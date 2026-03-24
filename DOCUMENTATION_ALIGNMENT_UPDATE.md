# NutriScan Official Documentation Alignment (March 2026)

This file lists the required edits to align `GEELEC 1 - Communications & AI.docx` with the **current codebase** in this repository.

## Scope

- Source references reviewed:
  - `nutriscan_model_report.docx` (project technical report)
  - `GEELEC 1 - Communications & AI.docx` (official submission document)
- Code references reviewed:
  - `model/trained_model.py`
  - `app.py`
  - UI reporting logic and calibration additions in `static/js/index.js` and `static/css/index.css`

## Executive Change Summary

| Area | Previous Documentation | Current Project Reality | Required Update |
|---|---|---|---|
| Core vs all-features design | Mixed references to 9/14 features with derived terms in production | Running app uses **7 core** and **12 all-features** schema | Update all architecture and feature-count sections to 7/12 |
| Derived features | `fried_index` and `fried_starchy` described as active production features | Not present in current serving schema | Move to "historical attempts" or remove from active model description |
| Calibration reporting | Minimal/no calibration section in official doc | App now reports Brier, ECE, Avg Confidence, Observed Positive + reliability curve | Add dedicated calibration-report subsection in methodology/results |
| CSV evaluation summary | Traditional confusion summary only | Includes custom total accuracy: `((TP - FP) / TP) * 100` per model | Add formula and interpretation note |
| Added sugar handling | Prior report discusses log1p from older variant | **Now implemented in training pipeline** for all-features model | Update preprocessing pipeline section to state active `log1p(added_sugar)` |
| Core-field handling | Older wording implies silent fill behavior | `/predict` endpoint rejects missing core fields (422) | Update API behavior section |
| Thresholding | Binary-only wording in some sections | Three-way verdict in app: Healthy / Borderline / Unhealthy with 0.60/0.40 cutoffs | Update decision-policy section |

## Required Revisions by Section (Official Doc)

## I. Introduction

### Replace/adjust

| Existing statement type | Replace with |
|---|---|
| "Model uses features including derived fried indicators as standard path" | "Production model uses 7 core nutrients for mandatory prediction and 5 optional nutrients when available (12 total for all-features path)." |
| "Two classes only in final output" | "System presents three verdicts in UI: Healthy, Borderline, Unhealthy using probability thresholds." |
| Any static metric claim without context | "State metrics as version-bound and tied to training snapshot date/model artifact." |

### Insert (recommended paragraph)

The deployed NutriScan app uses a dual-pipeline Logistic Regression setup: a 7-feature core model and a 12-feature all-features model. Predictions are presented with a three-level decision policy (Healthy, Borderline, Unhealthy) based on calibrated healthy probability. The CSV module now includes reliability reporting (Brier score, ECE, and calibration curve) to evaluate probability quality, not just class accuracy.

## II. Model Features and Contributors

### A. Feature Inventory (must be corrected)

| Category | Current Active Features |
|---|---|
| Core (required) | `calories`, `carbohydrates`, `sugar`, `fat`, `saturated_fat`, `sodium`, `protein` |
| Optional | `fiber`, `cholesterol`, `added_sugar`, `vitamin_c`, `omega3` |

### B. Remove from active-feature table

| Feature | Current Status | Documentation action |
|---|---|---|
| `fried_index` | Not in current training/serving schema | Remove from active tables; optionally mention under historical experiments |
| `fried_starchy` | Not in current training/serving schema | Remove from active tables; optionally mention under historical experiments |

### C. Coefficient interpretation wording update

Use this wording pattern:

- "Coefficient magnitudes are model-version specific and should be interpreted with preprocessing context (imputation, scaling, and optional transformations such as log1p on added_sugar)."
- Avoid hardcoding old coefficient values from previous variants unless re-extracted from current artifact.

## III. Algorithm and Pipeline

### Active preprocessing (current)

| Step | Core Model | All-Features Model | Notes |
|---|---|---|---|
| Imputation | Median | Median | Pipeline-based (`SimpleImputer`) |
| Log transform | No | **Yes: `log1p` on `added_sugar`** | Implemented via `FunctionTransformer` |
| Scaling | StandardScaler | StandardScaler | Applied after prior transforms |
| Classifier | Calibrated Logistic Regression | Calibrated Logistic Regression | `CalibratedClassifierCV(method='isotonic', cv=5)` |

### Decision policy (must be explicit)

| Condition on `P(Healthy)` | Verdict |
|---|---|
| `>= 0.60` | Healthy |
| `<= 0.40` | Unhealthy |
| `(0.40, 0.60)` | Borderline |

## IV. Evaluation and Reporting (new mandatory subsection)

Add a subsection named **"Calibration and Reliability Reporting"** with the following:

| Metric | Meaning | Where used |
|---|---|---|
| Brier Score | Mean squared error of predicted probabilities | CSV comparison summary |
| ECE (%) | Calibration gap between confidence and observed frequency | CSV comparison summary |
| Avg Confidence (%) | Mean predicted healthy probability (binned summary) | CSV comparison summary |
| Observed Positive (%) | Empirical healthy rate in calibration bins | CSV comparison summary |
| Reliability Curve | Predicted vs observed probability line graph | Classification report UI |

Also document this custom comparison metric:

| Metric | Formula | Note |
|---|---|---|
| Total Accuracy (%) | `((TP - FP) / TP) * 100` | Custom project metric; define separately from standard accuracy |

## V. API/Behavior Notes (must align with app)

| Endpoint | Current behavior | Required doc note |
|---|---|---|
| `/predict` | Rejects missing core fields with HTTP 422 | "Core nutrient fields are mandatory in manual mode." |
| `/predict-csv` | Allows missing cores via imputation, flags affected rows | "Batch mode reports imputed core warnings and per-model comparison summaries." |
| `/debug` | Exposes thresholds and feature schema counts | Useful for reproducibility appendix |

## VI. Problems Encountered / Learnings (refresh wording)

Keep lessons learned, but update technical wording:

- Replace "logic void" style narrative with:
  - "Feature skew and missingness handling materially affected coefficient stability."
  - "Calibration analysis was added to validate probability quality beyond top-line accuracy."
  - "Current implementation applies log1p compression to added_sugar in the all-features training path."

## VII. Suggested Text to Insert (ready-to-paste)

### Current model architecture

NutriScan currently deploys two calibrated Logistic Regression pipelines: a 7-feature core model and a 12-feature all-features model. The all-features pipeline applies median imputation, `log1p` transformation on `added_sugar`, standard scaling, and isotonic probability calibration. The UI reports both model verdicts and a calibration report that includes Brier score, ECE, and a reliability curve (predicted vs observed healthy probability).

### Interpretation warning

Coefficient values and feature attributions are version-dependent and must be interpreted together with preprocessing steps. Documentation should avoid reusing coefficients from older model variants unless regenerated from the current trained artifact.

## VIII. Checklist Before Submission

| Check | Status target |
|---|---|
| Feature counts and names match code (`7 core`, `12 all`) | Required |
| No derived features listed as active unless present in code | Required |
| Calibration section present (Brier, ECE, reliability curve) | Required |
| Decision thresholds documented (`0.60 / 0.40`) | Required |
| API behavior reflects core-field validation in manual mode | Required |
| Any metric values labeled with model/date/version | Required |

---

Prepared for immediate document synchronization with current NutriScan implementation.
