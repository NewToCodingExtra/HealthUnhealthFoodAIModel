from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from model.transforms import (
    BASE_CORE_FEATURES,
    DERIVED_FEATURES,
    OPTIONAL_FEATURES,
    CORE_FEATURES,
    ALL_FEATURES,
    compute_derived_features,
    log1p_added_sugar,
)

app = Flask(__name__)
CORS(app)

# ── Load trained model ────────────────────────────────────────────────────────
data      = joblib.load('trained_model.pkl')
pipelines = data['pipelines']       # full sklearn Pipelines (imputer+scaler+clf)
# Legacy scalar/imputer references kept for build_feature_contributions
scalers   = data['scalers']
imputer   = data['imputer']

# ── Canonical feature schema (must match trained_model.py and test scripts) ───
# Base core (7): always on nutrition label — required for every prediction
base_core_features = BASE_CORE_FEATURES
derived_features = DERIVED_FEATURES
optional_features = OPTIONAL_FEATURES
core_features = CORE_FEATURES
all_features = ALL_FEATURES

# ── HTML key → model feature name ─────────────────────────────────────────────
HTML_TO_MODEL = {feat: feat for feat in base_core_features + optional_features}


def build_feature_contributions(pipeline, X_df, feature_names, raw_values):
    """
    Return ALL features with their contribution weights and human-readable reasons.
    Sorted by absolute contribution (most impactful first).

    We derive the scaled array from the pipeline's preprocessor steps
    so contributions reflect the actual transformed values used for prediction.

    contrib = coef * scaled_value
      > 0 → pushing toward Healthy
      < 0 → pushing toward Unhealthy
    """
    # Apply every preprocessing step in pipeline order (except final classifier),
    # so contributions stay correct even if we add transforms like log1p.
    X_scaled = pipeline[:-1].transform(X_df)

    # Extract coefficients from CalibratedClassifierCV → first calibrator
    try:
        coef = pipeline.named_steps['clf'].calibrated_classifiers_[0].estimator.coef_[0]
    except AttributeError:
        coef = pipeline.named_steps['clf'].estimator.coef_[0]

    contrib = coef * X_scaled[0]
    ranked  = sorted(
        zip(feature_names, contrib, coef, raw_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    results = []
    for feat, val, c, raw in ranked:
        pushes_healthy = val > 0
        label = feat.replace("_", " ").title()

        if c > 0:
            reason = f"High {label} — good, more is healthier" if pushes_healthy \
                     else f"Low {label} — bad, more is healthier"
        else:
            reason = f"Low {label} — good, less is healthier" if pushes_healthy \
                     else f"High {label} — bad, less is healthier"

        results.append({
            "feature":   feat,
            "label":     label,
            "raw_value": round(float(raw), 4) if not np.isnan(raw) else None,
            "direction": "Healthy signal" if pushes_healthy else "Unhealthy signal",
            "reason":    reason,
            "weight":    round(float(val), 4)
        })

    return results


def normalize_expected_label(raw_value):
    """Map CSV expected_output values into Healthy/Unhealthy when possible."""
    if raw_value is None or pd.isna(raw_value):
        return None
    val = str(raw_value).strip().lower()
    if val == "":
        return None
    healthy_tokens   = {"healthy", "health", "h", "1", "true", "yes", "y", "good"}
    unhealthy_tokens = {"unhealthy", "unhealth", "u", "0", "false", "no", "n", "bad"}
    if val in healthy_tokens:
        return "Healthy"
    if val in unhealthy_tokens:
        return "Unhealthy"
    return None


def get_confusion_bucket(expected_label, predicted_label):
    """Return TP/TN/FP/FN for Healthy-vs-Unhealthy binary comparison."""
    if expected_label not in ("Healthy", "Unhealthy"):
        return None
    if predicted_label not in ("Healthy", "Unhealthy"):
        return None
    if predicted_label == "Healthy"   and expected_label == "Healthy":   return "TP"
    if predicted_label == "Unhealthy" and expected_label == "Unhealthy": return "TN"
    if predicted_label == "Healthy"   and expected_label == "Unhealthy": return "FP"
    return "FN"


def safe_total_accuracy(tp, fp):
    """
    Custom total-accuracy metric requested by UI:
    ((TP - FP) / TP) * 100
    """
    if tp <= 0:
        return None
    return round(((tp - fp) / tp) * 100, 2)


def compute_calibration_metrics(y_true, y_prob, bins=10):
    """
    Return lightweight calibration summary for UI reporting.
    """
    if len(y_true) == 0:
        return {
            "brier_score": None,
            "ece_percent": None,
            "avg_confidence_percent": None,
            "avg_observed_positive_percent": None,
            "sample_count": 0,
            "curve_points": [],
        }

    y_true_arr = np.array(y_true, dtype=float)
    y_prob_arr = np.array(y_prob, dtype=float)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(y_true_arr)
    observed = []
    confs = []
    curve_points = []

    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        # Include right edge on final bin only.
        if i == bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)

        if not np.any(mask):
            continue

        bin_true = y_true_arr[mask]
        bin_prob = y_prob_arr[mask]
        acc = float(np.mean(bin_true))
        conf = float(np.mean(bin_prob))
        weight = float(np.sum(mask)) / total
        ece += abs(acc - conf) * weight
        observed.append(acc)
        confs.append(conf)
        curve_points.append({
            "mean_predicted_percent": round(conf * 100, 2),
            "actual_positive_percent": round(acc * 100, 2),
            "count": int(np.sum(mask)),
        })

    avg_conf = float(np.mean(y_prob_arr))
    avg_obs = float(np.mean(y_true_arr))
    if observed:
        avg_obs = float(np.mean(observed))
    if confs:
        avg_conf = float(np.mean(confs))

    return {
        "brier_score": round(brier, 6),
        "ece_percent": round(ece * 100, 2),
        "avg_confidence_percent": round(avg_conf * 100, 2),
        "avg_observed_positive_percent": round(avg_obs * 100, 2),
        "sample_count": int(total),
        "curve_points": curve_points,
    }


# ── Verdict thresholds ────────────────────────────────────────────────────────
# These should be tuned from a validation precision/recall sweep
# (see threshold sweep output printed by trained_model.py).
# Current values are a reasonable starting point; adjust after reviewing sweep.
HEALTHY_THRESHOLD   = 0.60
UNHEALTHY_THRESHOLD = 0.40


def get_verdict(prob_healthy):
    if prob_healthy >= HEALTHY_THRESHOLD:
        return "Healthy",    True,  False
    elif prob_healthy <= UNHEALTHY_THRESHOLD:
        return "Unhealthy",  False, False
    else:
        return "Borderline", None,  True    # is_healthy=None means uncertain


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload   = request.get_json()
        food_name = payload.get('food_name', 'Unknown Food')
        nutrients = payload.get('nutrients', {})   # {html_key: value}

        # ── FIX (High): Reject missing core fields instead of silently filling 0 ─
        # Unknown ≠ zero for nutrition: missing calories is very different from
        # zero calories. Defaulting to 0 biases the model (e.g. 0 calories looks
        # like water to the model). Core fields must be provided.
        missing_cores = []
        core_vals = []
        for feat in base_core_features:
            val = nutrients.get(HTML_TO_MODEL[feat], None)
            if val is None:
                missing_cores.append(feat)
            else:
                core_vals.append(float(val))

        if missing_cores:
            return jsonify({
                "error": "missing_core_fields",
                "message": (
                    f"These required fields are missing: {', '.join(missing_cores)}. "
                    "Please fill in all core nutritional values before predicting."
                ),
                "missing_fields": missing_cores,
            }), 422

        # ── Build all-feature vector (NaN for unknown optionals) ──────────────
        # null from JS = user didn't enter value = truly unknown → imputer estimates
        # 0 from JS    = user entered zero       = genuinely none (e.g. no added sugar)
        carbohydrates = core_vals[base_core_features.index("carbohydrates")]
        fat = core_vals[base_core_features.index("fat")]
        fried_index, fried_starchy = compute_derived_features(carbohydrates, fat)
        core_model_vals = core_vals + [fried_index, fried_starchy]

        all_vals = core_model_vals.copy()
        for feat in optional_features:
            val = nutrients.get(HTML_TO_MODEL[feat], None)
            all_vals.append(np.nan if val is None else float(val))

        # ── Predict via full Pipelines ─────────────────────────────────────────
        X_core_df = pd.DataFrame([core_model_vals], columns=core_features)
        X_all_df  = pd.DataFrame([all_vals],  columns=all_features)

        # Fill missing core values with training median (from the all-features imputer)
        core_medians = dict(zip(all_features, imputer.statistics_))
        X_core_df = X_core_df.fillna({f: core_medians[f] for f in core_features})

        prob_core = pipelines['core'].predict_proba(X_core_df)[0]
        prob_all  = pipelines['all'].predict_proba(X_all_df)[0]

        core_label, core_is_healthy, core_borderline = get_verdict(float(prob_core[1]))
        all_label, all_is_healthy, all_borderline = get_verdict(float(prob_all[1]))

        models_disagree = (
            core_label != all_label
            and not (core_borderline or all_borderline)
        )

        return jsonify({
            "food_name": food_name,
            "core_model": {
                "label":          core_label,
                "is_healthy":     core_is_healthy,
                "is_borderline":  core_borderline,
                "prob_healthy":   round(float(prob_core[1]) * 100, 1),
                "prob_unhealthy": round(float(prob_core[0]) * 100, 1),
                "features":       build_feature_contributions(
                                      pipelines['core'], X_core_df,
                                      core_features, core_model_vals),
            },
            "all_model": {
                "label":          all_label,
                "is_healthy":     all_is_healthy,
                "is_borderline":  all_borderline,
                "prob_healthy":   round(float(prob_all[1]) * 100, 1),
                "prob_unhealthy": round(float(prob_all[0]) * 100, 1),
                "features":       build_feature_contributions(
                                      pipelines['all'], X_all_df,
                                      all_features, all_vals),
            },
            "models_disagree": models_disagree,
            "warning": (
                "⚠️ Models disagree — try entering optional features "
                "(fiber, cholesterol, added sugar, vitamin C, omega-3)"
            ) if models_disagree else None,
            "imputed_optionals": [
                f for f, v in zip(optional_features, all_vals[len(core_features):])
                if np.isnan(v)
            ],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/debug')
def debug():
    return jsonify({
        "core_features":        core_features,
        "base_core_features":   base_core_features,
        "derived_features":     derived_features,
        "optional_features":    optional_features,
        "core_model_n_features":  pipelines['core'].n_features_in_,
        "all_model_n_features":   pipelines['all'].n_features_in_,
        "healthy_threshold":    HEALTHY_THRESHOLD,
        "unhealthy_threshold":  UNHEALTHY_THRESHOLD,
    })


@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a .csv"}), 400

        df = pd.read_csv(file)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        name_col     = next((c for c in df.columns if c in ('name', 'food', 'food_name', 'item')), None)
        expected_col = "expected_output" if "expected_output" in df.columns else None

        results = []
        comparison_available = expected_col is not None

        # ── FIX (Medium): Track skipped counts per model independently ─────────
        # Old code incremented skipped_rows only when the core bucket was missing,
        # so all_model borderline cases were undercounted.
        confusion = {
            "core_model": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "all_model":  {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "comparable_rows": 0,
            "skipped_core":    0,   # rows where core_label was Borderline/unknown
            "skipped_all":     0,   # rows where all_label  was Borderline/unknown
        }
        y_true_binary = []   # Healthy=1, Unhealthy=0
        core_probs = []
        all_probs = []

        for idx, row in df.iterrows():
            food_name = str(row[name_col]) if name_col else f"Food #{idx+1}"

            # ── Core feature vector ───────────────────────────────────────────
            # FIX: Track which cores are missing; do NOT silently zero-fill.
            # For batch CSV we still need to produce a result per row, so we
            # impute missing cores via the training median and flag the row.
            core_vals  = []
            blank_cores = []
            for feat in base_core_features:
                if feat not in df.columns or pd.isna(row[feat]):
                    core_vals.append(np.nan)   # will be caught below
                    blank_cores.append(feat)
                else:
                    core_vals.append(float(row[feat]))

            # If any core is missing we can still run prediction with NaN,
            # but we record a warning on the row so the caller knows.
            carbs = core_vals[base_core_features.index("carbohydrates")]
            fat = core_vals[base_core_features.index("fat")]
            fried_index, fried_starchy = compute_derived_features(carbs, fat)
            core_model_vals = core_vals + [fried_index, fried_starchy]

            all_vals = core_model_vals.copy()
            for feat in optional_features:
                if feat not in df.columns or pd.isna(row.get(feat, np.nan)):
                    all_vals.append(np.nan)
                else:
                    all_vals.append(float(row[feat]))

            X_core_df = pd.DataFrame([core_model_vals], columns=core_features)
            X_all_df  = pd.DataFrame([all_vals],  columns=all_features)

            # Pipelines handle imputation internally
            prob_core = pipelines['core'].predict_proba(X_core_df)[0]
            prob_all  = pipelines['all'].predict_proba(X_all_df)[0]

            core_label, core_is_healthy, core_borderline = get_verdict(float(prob_core[1]))
            all_label, all_is_healthy, all_borderline = get_verdict(float(prob_all[1]))

            expected_label = normalize_expected_label(row[expected_col]) if expected_col else None
            core_match = expected_label == core_label if expected_label else None
            all_match  = expected_label == all_label  if expected_label else None

            if expected_label:
                confusion["comparable_rows"] += 1
                y_true_binary.append(1 if expected_label == "Healthy" else 0)
                core_probs.append(float(prob_core[1]))
                all_probs.append(float(prob_all[1]))

                core_bucket = get_confusion_bucket(expected_label, core_label)
                if core_bucket:
                    confusion["core_model"][core_bucket] += 1
                else:
                    confusion["skipped_core"] += 1   # borderline or unknown

                all_bucket = get_confusion_bucket(expected_label, all_label)
                if all_bucket:
                    confusion["all_model"][all_bucket] += 1
                else:
                    confusion["skipped_all"] += 1    # borderline or unknown

            results.append({
                "food_name":       food_name,
                "expected_output": expected_label,
                "core_model": {
                    "label":          core_label,
                    "is_healthy":     core_is_healthy,
                    "is_borderline":  core_borderline,
                    "prob_healthy":   round(float(prob_core[1]) * 100, 1),
                    "prob_unhealthy": round(float(prob_core[0]) * 100, 1),
                    "features":       build_feature_contributions(
                                          pipelines['core'], X_core_df,
                                          core_features, core_model_vals),
                },
                "all_model": {
                    "label":          all_label,
                    "is_healthy":     all_is_healthy,
                    "is_borderline":  all_borderline,
                    "prob_healthy":   round(float(prob_all[1]) * 100, 1),
                    "prob_unhealthy": round(float(prob_all[0]) * 100, 1),
                    "features":       build_feature_contributions(
                                          pipelines['all'], X_all_df,
                                          all_features, all_vals),
                },
                "comparison": {
                    "core_match": core_match,
                    "all_match":  all_match,
                },
                "blank_cores": blank_cores,
                "blank_core_warning": (
                    f"⚠️ Missing core fields (imputed from training median): "
                    f"{', '.join(blank_cores)}"
                ) if blank_cores else None,
            })

        if comparison_available:
            core_tp = confusion["core_model"]["TP"]
            core_fp = confusion["core_model"]["FP"]
            all_tp = confusion["all_model"]["TP"]
            all_fp = confusion["all_model"]["FP"]
            confusion["core_model"]["total_accuracy_percent"] = safe_total_accuracy(core_tp, core_fp)
            confusion["all_model"]["total_accuracy_percent"] = safe_total_accuracy(all_tp, all_fp)
            confusion["calibration_report"] = {
                "core_model": compute_calibration_metrics(y_true_binary, core_probs),
                "all_model": compute_calibration_metrics(y_true_binary, all_probs),
            }

        return jsonify({
            "results":              results,
            "comparison_available": comparison_available,
            "comparison_summary":   confusion if comparison_available else None,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)