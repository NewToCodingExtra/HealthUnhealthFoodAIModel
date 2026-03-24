from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# ── Load trained model ────────────────────────────────────────────────────────
data    = joblib.load('trained_model.pkl')
models  = data['models']
scalers = data['scalers']
imputer = data['imputer']
# Features that were log1p-transformed during training — same must be applied at inference.
log_transformed_features = data.get('log_transformed_features', [])
# Engineered features computed from core macros at prediction time.
engineered_features      = data.get('engineered_features', {})
# Calibration payload computed during training (reliability curve + Brier score).
calibration_payload    = data.get('calibration', None)

# ── Feature definitions (loaded from pkl to stay in sync with trained_model.py) ─
core_features     = data['core_features']
optional_features = data['optional_features']
all_features      = core_features + optional_features

# Raw nutrient inputs from the user (before engineering).
# These are the 7 base macros the user types in.
BASE_NUTRIENTS = ["calories","carbohydrates","sugar","fat","saturated_fat","sodium","protein"]

def compute_engineered(nutrient_vals):
    carbs = nutrient_vals.get('carbohydrates', 0.0) or 0.0
    fat = nutrient_vals.get('fat', 0.0) or 0.0
    sat_fat = nutrient_vals.get('saturated_fat', 0.0) or 0.0
    sodium = nutrient_vals.get('sodium', 0.0) or 0.0
    protein = nutrient_vals.get('protein', 0.0) or 0.0
    fiber_raw = nutrient_vals.get('fiber', np.nan)
    fiber = 0.0 if pd.isna(fiber_raw) else float(fiber_raw)
    added_sugar_raw = nutrient_vals.get('added_sugar', np.nan)

    nutrient_vals['fried_index_v2'] = float(
        np.log1p((carbs * fat) / 100.0) if (carbs >= 10 and fat >= 5) else 0.0
    )
    nutrient_vals['fried_starchy'] = float(carbs > 35 and fat > 8)
    nutrient_vals['has_carbs_and_fat'] = float(carbs > 2 and fat > 2)
    nutrient_vals['net_carbs'] = float(max(carbs - fiber, 0.0))
    nutrient_vals['sat_fat_protein_risk'] = float(
        sat_fat * (protein / 30.0) if sat_fat > 6 else 0.0
    )
    nutrient_vals['oil_quality'] = float(
        fat > 70
        and sodium < 10
        and sat_fat < 20
        and (not pd.isna(added_sugar_raw))
        and float(added_sugar_raw) == 0.0
    )
    nutrient_vals['sodium_protein_risk'] = float(
        (sodium / 300.0) * (1.0 + (protein / 30.0)) if sodium >= 450 else 0.0
    )
    nutrient_vals['refined_carb_density'] = float(carbs / (fiber + 1.0))
    nutrient_vals['whole_food_fat_protection'] = float(
        fat > 35
        and fiber > 4
        and sat_fat < 12
        and (not pd.isna(added_sugar_raw))
        and float(added_sugar_raw) <= float(np.log1p(6))
    )
    nutrient_vals['ultra_refined_carb_penalty'] = float(
        -1.0
        if (carbs > 40 and fiber < 3.5 and sodium > 200)
        else 0.0
    )
    nutrient_vals['lean_liquid_micronutrient_bonus'] = float(
        (nutrient_vals.get('vitamin_c', np.nan) > float(np.log1p(20)))
        and sodium < 20
        and fat < 1
        and sat_fat < 1
        and protein < 2
    )
    nutrient_vals['low_fiber_high_fat_protein_risk'] = float(
        protein > 18 and fat > 12 and fiber < 1
    )
    nutrient_vals['processed_refined_carb_salt_risk'] = float(
        carbs > 40
        and sodium > 350
        and fat < 6
        and protein < 11
        and (not pd.isna(added_sugar_raw))
        and float(added_sugar_raw) > float(np.log1p(1))
    )
    nutrient_vals['fried_protein_density_risk'] = float(
        (protein * fat) / (fiber + 1.0)
        if (protein > 18 and fat > 12 and sodium > 450)
        else 0.0
    )
    nutrient_vals['fatty_meat_risk'] = float(
        protein > 20 and fat > 15 and sat_fat > 5 and carbs < 5
    )
    nutrient_vals['refined_carb_sodium_low_fiber_strength'] = float(
        -1.0 * (carbs / (fiber + 1.0)) * (sodium / 500.0)
        if (carbs > 45 and fiber < 3 and sodium > 300)
        else 0.0
    )
    nutrient_vals['refined_carb_density_strong'] = float((carbs / (fiber + 1.0)) ** 2)
    return nutrient_vals

# ── Log-transform helper ──────────────────────────────────────────────────────
def apply_log_transforms(all_vals_list):
    """Apply log1p to any feature flagged in log_transformed_features.
    NaN values are first replaced with 0 for log-transformed features
    (log1p(0) = 0 = neutral), then the rest of the NaN values are left
    for the imputer to fill with the training median."""
    result = list(all_vals_list)
    for feat in log_transformed_features:
        if feat in all_features:
            idx = all_features.index(feat)
            val = result[idx]
            result[idx] = np.log1p(0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else val)
    return result


def apply_log_transforms_to_dict(nutrient_vals):
    """Apply the same log1p transforms used in training to a dict."""
    result = dict(nutrient_vals)
    for feat in log_transformed_features:
        if feat in result:
            val = result[feat]
            if val is not None and not pd.isna(val):
                transformed = float(np.log1p(val))
                if feat == "omega3":
                    transformed = float(np.log1p(np.clip(val, 0.0, 2.0)))
                elif feat == "vitamin_c":
                    transformed = float(np.log1p(np.clip(val, 0.0, 120.0)))
                result[feat] = transformed
    return result

# ── HTML key → model feature name (all match directly now) ───────────────────
HTML_TO_MODEL = {
    "calories":      "calories",
    "carbohydrates": "carbohydrates",
    "sugar":         "sugar",
    "fat":           "fat",
    "saturated_fat": "saturated_fat",
    "sodium":        "sodium",
    "protein":       "protein",
    "fiber":         "fiber",
    "cholesterol":   "cholesterol",
    "added_sugar":   "added_sugar",
    "vitamin_c":     "vitamin_c",
    "omega3":        "omega3",
}


def build_feature_contributions(model, X_scaled, feature_names, raw_values):
    """
    Return ALL features with their contribution weights and human-readable reasons.
    Sorted by absolute contribution (most impactful first).

    contrib = coef * scaled_value
      > 0 → pushing toward Healthy
      < 0 → pushing toward Unhealthy
    """
    coefs   = model.coef_[0]
    contrib = coefs * X_scaled[0]
    ranked  = sorted(
        zip(feature_names, contrib, coefs, raw_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    results = []
    for feat, val, coef, raw in ranked:
        pushes_healthy = val > 0
        label = feat.replace("_", " ").title()

        if coef > 0:
            if pushes_healthy:
                reason = f"High {label} — good, more is healthier"
            else:
                reason = f"Low {label} — bad, more is healthier"
        else:
            if pushes_healthy:
                reason = f"Low {label} — good, less is healthier"
            else:
                reason = f"High {label} — bad, less is healthier"

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

    healthy_tokens = {"healthy", "health", "h", "1", "true", "yes", "y", "good"}
    unhealthy_tokens = {"unhealthy", "unhealth", "u", "0", "false", "no", "n", "bad"}

    if val in healthy_tokens:
        return "Healthy"
    if val in unhealthy_tokens:
        return "Unhealthy"
    return None


def get_confusion_bucket(expected_label, predicted_label):
    """
    Return TP/TN/FP/FN for Healthy-vs-Unhealthy binary comparison.
    Borderline/unknown values return None.
    """
    if expected_label not in ("Healthy", "Unhealthy"):
        return None
    if predicted_label not in ("Healthy", "Unhealthy"):
        return None

    if predicted_label == "Healthy" and expected_label == "Healthy":
        return "TP"
    if predicted_label == "Unhealthy" and expected_label == "Unhealthy":
        return "TN"
    if predicted_label == "Healthy" and expected_label == "Unhealthy":
        return "FP"
    return "FN"


def safe_total_accuracy(tp, fp, tn, fn):
    """
    Total accuracy across both labels (same confusion matrix as the donut charts).

    Uses all four cells: correct = TP + TN, incorrect = FP + FN, then
    ((correct - incorrect) / correct) * 100.

    (Using only TP/FP from the Healthy-positive slice hid FN errors, e.g. one
    wrong Unhealthy prediction still showed 100%.)
    """
    total_correct = tp + tn
    if total_correct <= 0:
        return None
    total_incorrect = fp + fn
    return round(((total_correct - total_incorrect) / total_correct) * 100, 2)


def compute_calibration_metrics(y_true, y_prob, bins=8):
    """
    Return calibration summary + reliability curve points for UI.

    Brier uses all rows. ECE and the curve use quantile (equal-frequency) bins
    so each point aggregates enough labels to avoid 0/1 spikes from sparse
    equal-width bins on small CSVs.
    """
    if len(y_true) == 0:
        return {
            "brier_score": None,
            "ece_percent": None,
            "avg_confidence_percent": None,
            "avg_observed_positive_percent": None,
            "sample_count": 0,
            "curve_points": [],
            "binning": "quantile",
        }

    y_true_arr = np.array(y_true, dtype=float)
    y_prob_arr = np.array(y_prob, dtype=float)
    brier = float(((y_prob_arr - y_true_arr) ** 2).mean())

    n = len(y_true_arr)
    n_bins = min(bins, max(3, n // 5))
    n_bins = max(3, min(n_bins, n))

    order = np.argsort(y_prob_arr)
    sorted_true = y_true_arr[order]
    sorted_prob = y_prob_arr[order]

    ece = 0.0
    curve_points = []

    for b in range(n_bins):
        lo = int(b * n / n_bins)
        hi = int((b + 1) * n / n_bins) if b < n_bins - 1 else n
        if lo >= hi:
            continue
        bin_true = sorted_true[lo:hi]
        bin_prob = sorted_prob[lo:hi]
        acc = float(np.mean(bin_true))
        conf = float(np.mean(bin_prob))
        cnt = hi - lo
        ece += abs(acc - conf) * (float(cnt) / n)
        curve_points.append({
            "mean_predicted_percent": round(conf * 100, 2),
            "actual_positive_percent": round(acc * 100, 2),
            "count": int(cnt),
        })

    curve_points.sort(key=lambda p: p["mean_predicted_percent"])

    return {
        "brier_score": round(brier, 6),
        "ece_percent": round(ece * 100, 2),
        "avg_confidence_percent": round(float(np.mean(y_prob_arr)) * 100, 2),
        "avg_observed_positive_percent": round(float(np.mean(y_true_arr)) * 100, 2),
        "sample_count": int(n),
        "curve_points": curve_points,
        "binning": "quantile",
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/calibration', methods=['GET'])
def calibration():
    if not calibration_payload:
        return jsonify({"error": "Calibration data not found in trained_model.pkl"}), 404
    return jsonify(calibration_payload)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload    = request.get_json()
        food_name  = payload.get('food_name', 'Unknown Food')
        nutrients  = payload.get('nutrients', {})   # {html_key: value}

        # ── Build core feature vector ─────────────────────────────────────────
        # Core features must always have a real value.
        # null/None means the user left it blank — treat as 0 with a warning.
        # 0 is valid (e.g. cholesterol=0 for plant-based foods).

        # First collect the raw base nutrients from the user
        raw_nutrients = {}
        blank_cores   = []
        for feat in BASE_NUTRIENTS:
            html_key = next((k for k, v in HTML_TO_MODEL.items() if v == feat), None)
            val = nutrients.get(html_key, None) if html_key else None
            if val is None:
                raw_nutrients[feat] = 0.0
                blank_cores.append(feat)
            else:
                raw_nutrients[feat] = float(val)

        optional_nutrients = {}
        for feat in optional_features:
            html_key = next((k for k, v in HTML_TO_MODEL.items() if v == feat), None)
            if html_key and html_key in nutrients:
                val = nutrients[html_key]
                optional_nutrients[feat] = np.nan if val is None else float(val)
            else:
                optional_nutrients[feat] = np.nan

        merged_nutrients = {**raw_nutrients, **optional_nutrients}
        merged_nutrients = apply_log_transforms_to_dict(merged_nutrients)
        compute_engineered(merged_nutrients)

        # Build the core vector in the order core_features expects
        core_vals = [merged_nutrients.get(f, 0.0) for f in core_features]

        # ── Build all-feature vector (NaN for unknown optionals) ──────────────
        all_vals = [merged_nutrients.get(f, np.nan) for f in all_features]

        # ── Scale & predict ───────────────────────────────────────────────────
        X_core_df = pd.DataFrame([core_vals], columns=core_features)
        X_all_df  = pd.DataFrame([all_vals],  columns=all_features)

        X_core_scaled = scalers['core'].transform(X_core_df)
        X_all_imputed = imputer.transform(X_all_df)
        X_all_scaled  = scalers['all'].transform(X_all_imputed)

        prob_core = models['core'].predict_proba(X_core_scaled)[0]
        prob_all  = models['all'].predict_proba(X_all_scaled)[0]

        # ── Rule-based override (last-mile risk guard) ─────────────────────
        # White-bread-like refined carbs can still be predicted as Healthy
        # with very high confidence. When the nutrient profile matches a
        # classic refined-carb + low-fiber + high-sodium pattern, we clamp
        # the probability and force an Unhealthy verdict.
        fiber_val_for_rule = merged_nutrients.get('fiber', np.nan)
        fiber_val_for_rule = 0.0 if pd.isna(fiber_val_for_rule) else float(fiber_val_for_rule)
        refined_carb_risk_guard = (
            merged_nutrients.get('carbohydrates', 0.0) > 45
            and fiber_val_for_rule < 3
            and merged_nutrients.get('sodium', 0.0) > 450
            and merged_nutrients.get('saturated_fat', 0.0) < 1.0
        )
        rule_override = False
        if refined_carb_risk_guard:
            # predict_proba order is [P(Unhealthy), P(Healthy)]
            prob_core = np.array([0.80, 0.20], dtype=float)
            prob_all  = np.array([0.80, 0.20], dtype=float)
            rule_override = True

        # ── Verdict with borderline zone ──────────────────────────────────────
        # Instead of a hard 0.5 cutoff, we use a confidence band:
        #   >= 0.60 → Healthy
        #   <= 0.40 → Unhealthy
        #   between → Borderline (model is uncertain, avoid a false verdict)
        HEALTHY_THRESHOLD   = 0.60
        UNHEALTHY_THRESHOLD = 0.40

        def get_verdict(prob_healthy):
            if prob_healthy >= HEALTHY_THRESHOLD:
                return "Healthy",    True,  False
            elif prob_healthy <= UNHEALTHY_THRESHOLD:
                return "Unhealthy",  False, False
            else:
                return "Borderline", None,  True   # is_healthy=None means uncertain

        core_label, core_is_healthy, core_borderline = get_verdict(float(prob_core[1]))
        all_label,  all_is_healthy,  all_borderline  = get_verdict(float(prob_all[1]))

        models_disagree = core_label != all_label and not (core_borderline or all_borderline)
        if rule_override:
            models_disagree = False

        return jsonify({
            "food_name": food_name,
            "core_model": {
                "label":           core_label,
                "is_healthy":      core_is_healthy,
                "is_borderline":   core_borderline,
                "prob_healthy":    round(float(prob_core[1]) * 100, 1),
                "prob_unhealthy":  round(float(prob_core[0]) * 100, 1),
                "features":        build_feature_contributions(models['core'], X_core_scaled, core_features, core_vals),
            },
            "all_model": {
                "label":           all_label,
                "is_healthy":      all_is_healthy,
                "is_borderline":   all_borderline,
                "prob_healthy":    round(float(prob_all[1]) * 100, 1),
                "prob_unhealthy":  round(float(prob_all[0]) * 100, 1),
                "features":        build_feature_contributions(models['all'], X_all_scaled, all_features, all_vals),
            },
            "models_disagree": models_disagree,
            "warning": (
                "⚠️ Rule override applied: refined-carb + low-fiber + high-sodium profile." if rule_override
                else ("⚠️ Models disagree — try entering optional features (added sugar, vitamin C, omega-3)" if models_disagree else None)
            ),
            "blank_core_warning": f"⚠️ These core fields were blank and defaulted to 0: {', '.join(blank_cores)}" if blank_cores else None,
            "imputed_optionals": [f for f, v in zip(optional_features, all_vals[len(core_features):]) if np.isnan(v)]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug')
def debug():
    return jsonify({
        "core_features": core_features,
        "optional_features": optional_features,
        "core_model_n_features": models['core'].n_features_in_,
        "all_model_n_features":  models['all'].n_features_in_,
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

        # Normalise column names: lowercase + strip spaces
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Detect food name column
        name_col = next((c for c in df.columns if c in ('name','food','food_name','item')), None)
        expected_col = "expected_output" if "expected_output" in df.columns else None

        # Check which optional columns actually exist in the CSV
        # If the whole column is missing → entire column is NaN → imputer handles it
        # If the column exists but a cell is blank → that cell is NaN → imputer handles it
        # If the column exists and has a value → use it as-is (0 = genuinely zero)
        optional_cols_present = [f for f in optional_features if f in df.columns]
        optional_cols_missing = [f for f in optional_features if f not in df.columns]

        results = []
        comparison_available = expected_col is not None
        confusion = {
            "core_model": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "all_model": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "comparable_rows": 0,
            "skipped_rows": 0,
        }
        y_true_binary = []
        core_probs = []
        all_probs = []
        for idx, row in df.iterrows():
            food_name = str(row[name_col]) if name_col else f"Food #{idx+1}"

            # ── Core feature vector ───────────────────────────────────────────
            # Collect base nutrients from CSV row, then compute engineered features.
            raw_row = {}
            blank_cores = []
            for feat in BASE_NUTRIENTS:
                if feat not in df.columns:
                    raw_row[feat] = 0.0
                    blank_cores.append(feat)
                else:
                    val = row[feat]
                    if pd.isna(val):
                        raw_row[feat] = 0.0
                        blank_cores.append(feat)
                    else:
                        raw_row[feat] = float(val)

            optional_row = {}
            for feat in optional_features:
                if feat not in df.columns:
                    optional_row[feat] = np.nan
                else:
                    val = row[feat]
                    optional_row[feat] = np.nan if pd.isna(val) else float(val)

            merged_row = {**raw_row, **optional_row}
            merged_row = apply_log_transforms_to_dict(merged_row)
            compute_engineered(merged_row)
            core_vals = [merged_row.get(f, 0.0) for f in core_features]

            # ── All-feature vector ────────────────────────────────────────────
            all_vals = [merged_row.get(f, np.nan) for f in all_features]

            X_core_df = pd.DataFrame([core_vals], columns=core_features)
            X_all_df  = pd.DataFrame([all_vals],  columns=all_features)

            X_core_scaled = scalers['core'].transform(X_core_df)
            X_all_imputed = imputer.transform(X_all_df)
            X_all_scaled  = scalers['all'].transform(X_all_imputed)

            prob_core = models['core'].predict_proba(X_core_scaled)[0]
            prob_all  = models['all'].predict_proba(X_all_scaled)[0]

            # ── Rule-based override (last-mile risk guard) ──────────────
            fiber_val_for_rule = merged_row.get('fiber', np.nan)
            fiber_val_for_rule = 0.0 if pd.isna(fiber_val_for_rule) else float(fiber_val_for_rule)
            refined_carb_risk_guard = (
                merged_row.get('carbohydrates', 0.0) > 45
                and fiber_val_for_rule < 3
                and merged_row.get('sodium', 0.0) > 450
                and merged_row.get('saturated_fat', 0.0) < 1.0
            )
            rule_override = False
            if refined_carb_risk_guard:
                prob_core = np.array([0.80, 0.20], dtype=float)
                prob_all  = np.array([0.80, 0.20], dtype=float)
                rule_override = True

            HEALTHY_THRESHOLD   = 0.60
            UNHEALTHY_THRESHOLD = 0.40

            def get_verdict(p):
                if p >= HEALTHY_THRESHOLD:   return "Healthy",    True,  False
                elif p <= UNHEALTHY_THRESHOLD: return "Unhealthy", False, False
                else:                          return "Borderline", None,  True

            core_label, core_is_healthy, core_borderline = get_verdict(float(prob_core[1]))
            all_label,  all_is_healthy,  all_borderline  = get_verdict(float(prob_all[1]))

            expected_label = normalize_expected_label(row[expected_col]) if expected_col else None
            core_match = expected_label == core_label if expected_label else None
            all_match = expected_label == all_label if expected_label else None

            if expected_label:
                confusion["comparable_rows"] += 1
                y_true_binary.append(1 if expected_label == "Healthy" else 0)
                core_probs.append(float(prob_core[1]))
                all_probs.append(float(prob_all[1]))

                core_bucket = get_confusion_bucket(expected_label, core_label)
                if core_bucket:
                    confusion["core_model"][core_bucket] += 1
                else:
                    confusion["skipped_rows"] += 1

                all_bucket = get_confusion_bucket(expected_label, all_label)
                if all_bucket:
                    confusion["all_model"][all_bucket] += 1

            # Keep UI consistent with the rule guard even when comparing.
            # (No extra warning field in CSV output; `warning` is for manual view.)

            results.append({
                "food_name":  food_name,
                "expected_output": expected_label,
                "core_model": {
                    "label":         core_label,
                    "is_healthy":    core_is_healthy,
                    "is_borderline": core_borderline,
                    "prob_healthy":  round(float(prob_core[1]) * 100, 1),
                    "prob_unhealthy":round(float(prob_core[0]) * 100, 1),
                    "features":      build_feature_contributions(models['core'], X_core_scaled, core_features, core_vals),
                },
                "all_model": {
                    "label":         all_label,
                    "is_healthy":    all_is_healthy,
                    "is_borderline": all_borderline,
                    "prob_healthy":  round(float(prob_all[1]) * 100, 1),
                    "prob_unhealthy":round(float(prob_all[0]) * 100, 1),
                    "features":      build_feature_contributions(models['all'], X_all_scaled, all_features, all_vals),
                },
                "comparison": {
                    "core_match": core_match,
                    "all_match": all_match,
                },
                "blank_cores": blank_cores,
            })

        if comparison_available:
            cm_core = confusion["core_model"]
            cm_all = confusion["all_model"]
            confusion["core_model"]["total_accuracy_percent"] = safe_total_accuracy(
                cm_core["TP"], cm_core["FP"], cm_core["TN"], cm_core["FN"]
            )
            confusion["all_model"]["total_accuracy_percent"] = safe_total_accuracy(
                cm_all["TP"], cm_all["FP"], cm_all["TN"], cm_all["FN"]
            )
            confusion["calibration_report"] = {
                "core_model": compute_calibration_metrics(y_true_binary, core_probs),
                "all_model": compute_calibration_metrics(y_true_binary, all_probs),
            }

        return jsonify({
            "results": results,
            "comparison_available": comparison_available,
            "comparison_summary": confusion if comparison_available else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)