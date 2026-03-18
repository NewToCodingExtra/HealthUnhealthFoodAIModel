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

# ── Feature definitions (must match trained_model.py) ────────────────────────
# Core (7): on every nutrition label — always required
# Optional (5): not present in all foods — imputed if missing
core_features = [
    "calories", "carbohydrates", "sugar", "fat",
    "saturated_fat", "sodium", "protein",
]
optional_features = [
    "fiber",        # absent/zero in many meat & processed foods
    "cholesterol",  # absent in plant-based foods
    "added_sugar",  # strongest unhealthy signal
    "vitamin_c",    # strong healthy signal
    "omega3",       # healthy fat signal
]
all_features = core_features + optional_features

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


def top_features(model, X_scaled, feature_names, raw_values, top_n=3):
    """
    Return top contributing features with clear human-readable reasons.

    contrib = coef * scaled_value
      > 0 → pushing toward Healthy
      < 0 → pushing toward Unhealthy

    coef > 0 means "more = healthier" (fiber, protein, omega3, vitamin_c)
    coef < 0 means "less = healthier" (sugar, sodium, calories, etc.)

    So:
      fiber (coef+), raw=0  → contrib < 0 → Low fiber — bad, more is better
      fiber (coef+), raw=10 → contrib > 0 → High fiber — good, more is better
      sodium (coef-), raw=800 → contrib < 0 → High sodium — bad, less is better
      sodium (coef-), raw=1   → contrib > 0 → Low sodium — good, less is better
    """
    coefs   = model.coef_[0]
    contrib = coefs * X_scaled[0]
    ranked  = sorted(
        zip(feature_names, contrib, coefs, raw_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    results = []
    for feat, val, coef, raw in ranked[:top_n]:
        pushes_healthy = val > 0
        label = feat.replace("_", " ").title()

        if coef > 0:
            # more = healthier (fiber, protein, omega3, vitamin_c)
            if pushes_healthy:
                reason = f"High {label} — good, more is healthier"
            else:
                reason = f"Low {label} — bad, more is healthier"
        else:
            # less = healthier (sugar, sodium, calories, fat, cholesterol, etc.)
            if pushes_healthy:
                reason = f"Low {label} — good, less is healthier"
            else:
                reason = f"High {label} — bad, less is healthier"

        results.append({
            "feature":   feat,
            "direction": "Healthy signal" if pushes_healthy else "Unhealthy signal",
            "reason":    reason,
            "value":     round(float(val), 4)
        })

    return results


@app.route('/')
def home():
    return render_template('index.html')


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
        core_vals = []
        blank_cores = []
        for feat in core_features:
            html_key = next((k for k, v in HTML_TO_MODEL.items() if v == feat), None)
            val = nutrients.get(html_key, None) if html_key else None
            if val is None:
                core_vals.append(0.0)   # fallback: treat blank as 0
                blank_cores.append(feat)
            else:
                core_vals.append(float(val))

        # ── Build all-feature vector (NaN for unknown optionals) ──────────────
        # null from JS = user didn't enter value = truly unknown → imputer estimates
        # 0 from JS    = user entered zero       = genuinely none (e.g. no added sugar)
        all_vals = core_vals.copy()
        for feat in optional_features:
            html_key = next((k for k, v in HTML_TO_MODEL.items() if v == feat), None)
            if html_key and html_key in nutrients:
                val = nutrients[html_key]
                all_vals.append(np.nan if val is None else float(val))
            else:
                all_vals.append(np.nan)   # not in payload at all → imputer estimates

        # ── Scale & predict ───────────────────────────────────────────────────
        X_core_df = pd.DataFrame([core_vals], columns=core_features)
        X_all_df  = pd.DataFrame([all_vals],  columns=all_features)

        X_core_scaled = scalers['core'].transform(X_core_df)
        X_all_imputed = imputer.transform(X_all_df)
        X_all_scaled  = scalers['all'].transform(X_all_imputed)

        prob_core = models['core'].predict_proba(X_core_scaled)[0]
        prob_all  = models['all'].predict_proba(X_all_scaled)[0]

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

        return jsonify({
            "food_name": food_name,
            "core_model": {
                "label":           core_label,
                "is_healthy":      core_is_healthy,
                "is_borderline":   core_borderline,
                "prob_healthy":    round(float(prob_core[1]) * 100, 1),
                "prob_unhealthy":  round(float(prob_core[0]) * 100, 1),
                "top_features":    top_features(models['core'], X_core_scaled, core_features, core_vals),
            },
            "all_model": {
                "label":           all_label,
                "is_healthy":      all_is_healthy,
                "is_borderline":   all_borderline,
                "prob_healthy":    round(float(prob_all[1]) * 100, 1),
                "prob_unhealthy":  round(float(prob_all[0]) * 100, 1),
                "top_features":    top_features(models['all'], X_all_scaled, all_features, all_vals),
            },
            "models_disagree": models_disagree,
            "warning": "⚠️ Models disagree — try entering optional features (added sugar, vitamin C, omega-3)" if models_disagree else None,
            "blank_core_warning": f"⚠️ These core fields were blank and defaulted to 0: {', '.join(blank_cores)}" if blank_cores else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)