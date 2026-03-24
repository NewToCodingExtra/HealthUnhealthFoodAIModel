import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transforms import (
    BASE_CORE_FEATURES,
    DERIVED_FEATURES,
    OPTIONAL_FEATURES,
    CORE_FEATURES,
    ALL_FEATURES,
    add_derived_features,
    log1p_added_sugar,
)

base_core_features = BASE_CORE_FEATURES
derived_features = DERIVED_FEATURES
optional_features = OPTIONAL_FEATURES
core_features = CORE_FEATURES
all_features = ALL_FEATURES


np.random.seed(42)

df = pd.read_csv('nutrition_data_125k.csv')

print(f"Dataset loaded: {len(df)} rows")

n_healthy   = (df['health_label'] == 'Healthy').sum()
n_unhealthy = (df['health_label'] == 'Unhealthy').sum()
print(f"Labels: Healthy={n_healthy} ({100*n_healthy/len(df):.1f}%)  "
      f"Unhealthy={n_unhealthy} ({100*n_unhealthy/len(df):.1f}%)")

encoder = LabelEncoder()
df['health_label'] = encoder.fit_transform(df['health_label'])
# LabelEncoder sorts alphabetically: Healthy=0, Unhealthy=1
# Flip so Healthy=1 and Unhealthy=0 (consistent with app/test display)
df['health_label'] = 1 - df['health_label']

df_with_derived = add_derived_features(df)
X_core = df_with_derived[core_features]
X_all  = df_with_derived[all_features]
y      = df['health_label']

# ── FIX (High): Split BEFORE fitting any preprocessors ───────────────────────
# Fitting imputer/scaler on the full dataset before the split leaks test
# statistics into training, inflating reported accuracy.
# Correct approach: split first, then fit only on train data.
X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(
    X_core, y, test_size=0.2, random_state=42, stratify=y
)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# ── Build sklearn Pipelines (preprocessor fitted on train only) ───────────────
# Core pipeline: StandardScaler only (no NaNs expected in core)
pipe_core = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler",  StandardScaler()),
    ("clf",     CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000, class_weight='balanced', C=0.3),
                    cv=5, method='isotonic')),
])

# All-features pipeline: median imputation first, then scaling
pipe_all = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("log1p_added_sugar", FunctionTransformer(log1p_added_sugar, validate=False)),
    ("scaler",  StandardScaler()),
    ("clf",     CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1),
                    cv=5, method='isotonic')),
])

# fit_transform on train; transform-only on test (no leakage)
pipe_core.fit(X_train_core, y_train_core)
pipe_all.fit(X_train_all,  y_train_all)

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC    : {average_precision_score(y_test, y_prob):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # ── Per-class F1 from classification report ───────────────────────────────
    from sklearn.metrics import f1_score
    f1_healthy   = f1_score(y_test, y_pred, pos_label=1)
    f1_unhealthy = f1_score(y_test, y_pred, pos_label=0)
    print(f"F1 (Healthy)   : {f1_healthy:.4f}")
    print(f"F1 (Unhealthy) : {f1_unhealthy:.4f}")

    # ── Threshold sweep: show precision/recall at common thresholds ───────────
    print("\nThreshold sweep (healthy class):")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    from sklearn.metrics import precision_score, recall_score
    for t in np.arange(0.30, 0.75, 0.05):
        y_at_t = (y_prob >= t).astype(int)
        if y_at_t.sum() == 0:
            continue
        p = precision_score(y_test, y_at_t, zero_division=0)
        r = recall_score(y_test, y_at_t, zero_division=0)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{t:>10.2f} {p:>10.4f} {r:>8.4f} {f:>8.4f}")

    return y_prob

y_prob_core = evaluate("Core Model (9 features)", pipe_core, X_test_core, y_test_core)
y_prob_all  = evaluate("All-Features Model (14 features)", pipe_all, X_test_all, y_test_all)

# ── Feature importance (from inner LogisticRegression) ───────────────────────
def print_feature_importance(pipe, feature_names, model_name):
    # CalibratedClassifierCV wraps the base estimator
    # Access coefficients from the first calibrated estimator's base estimator
    try:
        coef = pipe.named_steps['clf'].calibrated_classifiers_[0].estimator.coef_[0]
    except AttributeError:
        coef = pipe.named_steps['clf'].estimator.coef_[0]
    importance = pd.DataFrame(
        {'Healthy': coef, 'Unhealthy': -coef},
        index=feature_names,
    )
    print(f"\nFeature importance ({model_name}):\n{importance.to_string()}")

print_feature_importance(pipe_core, core_features, "Core Model")
print_feature_importance(pipe_all, all_features, "All-Features Model")

# ── Calibration plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_prob, y_test, name in [
    (axes[0], y_prob_core, y_test_core, "Core Model"),
    (axes[1], y_prob_all,  y_test_all,  "All-Features Model"),
]:
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
    ax.set_title(f"Calibration — {name}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
plt.tight_layout()
plt.savefig('calibration_plots.png', dpi=150)
plt.close()
print("\nCalibration plots saved -> calibration_plots.png")

# ── Save artefacts ────────────────────────────────────────────────────────────
# Save the full Pipeline objects so app.py can call .transform() correctly
# without needing separate scaler/imputer references.
joblib.dump({
    'pipelines':        {'core': pipe_core, 'all': pipe_all},
    # Legacy keys kept for backward-compatibility with existing app.py
    # (app.py will be updated to use pipelines directly)
    'models':           {'core': pipe_core, 'all': pipe_all},
    'scalers':          {
        'core': pipe_core.named_steps['scaler'],
        'all':  pipe_all.named_steps['scaler'],
    },
    'imputer':           pipe_all.named_steps['imputer'],
    'core_features':     core_features,
    'base_core_features': base_core_features,
    'derived_features':   derived_features,
    'optional_features': optional_features,
    'all_features':      all_features,
}, 'trained_model.pkl')

print("\nModel saved -> trained_model.pkl")