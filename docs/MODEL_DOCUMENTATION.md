# NutriScan Model Documentation (Table Format)

## 1) Model Summary

| Item | Value |
|---|---|
| Problem type | Binary classification (`Healthy` vs `Unhealthy`) |
| Algorithm | Logistic Regression |
| Architecture | Two models: `core` + `all` |
| Core model purpose | Stable predictions when only label-common nutrients are available |
| All model purpose | Higher precision when optional nutrients are provided |
| Scaling | `StandardScaler` |
| Missing values | `SimpleImputer(strategy='median')` |
| Special imputation rule | `added_sugar` imputed to `log1p(0.5)` |
| Probability thresholds | `>=0.60 Healthy`, `<=0.40 Unhealthy`, else `Borderline` |
| Explainability | Per-feature contribution (`coefficient * scaled_value`) |

## 2) End-to-End Pipeline

| Step | What happens | Why it matters |
|---|---|---|
| 1 | Read raw nutrients | Base input from UI/CSV |
| 2 | Apply log transforms (`added_sugar`, `omega3`, `vitamin_c`) | Reduces skew from long-tail nutrient values |
| 3 | Compute engineered features | Captures interactions linear model cannot learn directly |
| 4 | Build ordered feature vectors | Ensures train/inference feature consistency |
| 5 | Impute missing optional values | Keeps all-model robust with incomplete inputs |
| 6 | Scale features | Makes coefficients comparable and stable |
| 7 | Predict probabilities | Produces `P(Healthy)` and `P(Unhealthy)` |
| 8 | Apply confidence band | Avoids forcing uncertain samples into hard labels |

## 3) Base Features

| Feature | Group | Typical signal role | Why included |
|---|---|---|---|
| `calories` | Core | Context-dependent | Captures energy density |
| `carbohydrates` | Core | Context-dependent | Broad carb load before quality adjustments |
| `sugar` | Core | Usually unhealthy | Captures simple sugar pressure |
| `fat` | Core | Context-dependent | Needed for fat-risk and fat-quality interactions |
| `saturated_fat` | Core | Usually unhealthy | Strong marker for fatty processed/animal profiles |
| `sodium` | Core | Strong unhealthy | Strong processed/salty-food risk |
| `protein` | Core | Usually healthy | Needed for protein-density and risk balancing |
| `fiber` | Optional | Strong healthy | Distinguishes whole-food vs refined-carb patterns |
| `cholesterol` | Optional | Weak/moderate | Secondary context for animal-based foods |
| `added_sugar` | Optional | Strong unhealthy | Key processed-sugar signal |
| `vitamin_c` | Optional | Usually healthy | Micronutrient context, especially produce/liquids |
| `omega3` | Optional | Usually healthy context | Fat-quality context for seafood/seeds/nuts |

## 4) Transformations and Imputation

| Component | Rule | Purpose |
|---|---|---|
| `added_sugar` transform | `log1p(added_sugar)` | Reduces heavy right-tail skew |
| `omega3` transform | `log1p(clip(omega3, 0, 2.0))` | Prevents extreme omega3 outliers dominating |
| `vitamin_c` transform | `log1p(clip(vitamin_c, 0, 120.0))` | Stabilizes very high vitamin C values |
| Optional missing values | `SimpleImputer(strategy='median')` | Preserves all-model operation with partial input |
| `added_sugar` imputer override | `statistics_[idx] = log1p(0.5)` | Prevents missing sugar being treated too optimistically |

## 5) Engineered Features Catalog

| Feature | Formula (simplified) | Main objective | Helps fix |
|---|---|---|---|
| `fried_index_v2` | `log1p((carbs*fat)/100)` if `carbs>=10` and `fat>=5` else `0` | Continuous “fried/starchy density” without the old zero-carb boost | Prevents zero-carb/zero-fat false Healthy |
| `fried_starchy` | `1` if `carbs>35` and `fat>8` else `0` | Hard fried-starchy trigger | Fries/chips-like under-penalization |
| `has_carbs_and_fat` | `1` if `carbs>2` and `fat>2` else `0` | Enables the interaction region | Interaction noise control |
| `net_carbs` | `max(carbohydrates - fiber, 0)` | Fiber-adjusted carb load | Refined vs whole carb confusion |
| `sat_fat_protein_risk` | `saturated_fat*(protein/30)` when `saturated_fat>6` else `0` | “Fatty meat risk” interaction | Lamb/chop patterns |
| `oil_quality` | `1` if `fat>70`, `sodium<10`, `saturated_fat<20`, and `added_sugar==0` (log space) else `0` | Protects healthy oils | Olive oil false Unhealthy |
| `sodium_protein_risk` | `(sodium/300)*(1+protein/30)` when `sodium>=450` else `0` | Salty + protein processed-food penalty | White-bread-like and salty processed profiles |
| `refined_carb_density` | `carbohydrates/(fiber+1)` | Refined-carb pressure ratio | Refined carb pressure |
| `refined_carb_density_strong` | `(carbohydrates/(fiber+1))^2` | Stronger refined-carb separation | Helps when the simple ratio is too weak |
| `ultra_refined_carb_penalty` | `-1` if (`carbs>40` and `fiber<3.5` and `sodium>200`) else `0` | Refined-carb penalty coded as a negative-valued indicator | Dataset-specific ultra-refined contexts |
| `refined_carb_sodium_low_fiber_strength` | `- (carbs/(fiber+1))*(sodium/500)` if (`carbs>45` and `fiber<3` and `sodium>300`) else `0` | Continuous refined-carb + salty + low-fiber penalty | Strengthens refined-carb risk in salty profiles |
| `whole_food_fat_protection` | `1` if `fat>35`, `fiber>4`, `saturated_fat<12`, and `added_sugar<=log1p(6)` else `0` | Protects whole-food fats | Nuts/seeds/whole fats |
| `processed_refined_carb_salt_risk` | `1` if (`carbs>40` and `sodium>350` and `fat<6` and `protein<11` and `added_sugar>log1p(1)`) else `0` | Refined-carb + salt + low-fat/low-protein risk | Processed/bread-like refined carbs |
| `lean_liquid_micronutrient_bonus` | `1` if (`vitamin_c > log1p(20)` and `sodium<20` and `fat<1` and `saturated_fat<1` and `protein<2`) else `0` | Bonus for micronutrient-rich low-fat liquids | Juice-like liquids |
| `low_fiber_high_fat_protein_risk` | `1` if (`protein>18` and `fat>12` and `fiber<1`) else `0` | Penalizes dense low-fiber fatty protein foods | Fried chicken patterns |
| `fried_protein_density_risk` | `(protein*fat)/(fiber+1)` if (`protein>18` and `fat>12` and `sodium>450`) else `0` | Fried/dense salty protein penalty | High-protein fried-like meals |
| `fatty_meat_risk` | `1` if (`protein>20` and `fat>15` and `saturated_fat>5` and `carbohydrates<5`) else `0` | Fatty meat/processed meat penalty | Fatty meat profiles |

## 6) Why Logistic Regression for this project

| Advantage | Practical impact |
|---|---|
| Fast training/inference | Easy retraining and low-latency Flask predictions |
| Explainable coefficients | Transparent feature impact for users and debugging |
| Stable with scaling | Reliable behavior across nutrients with different units |
| Works well with feature engineering | Lets domain rules drive high-value corrections |

## 7) Prediction Decision Logic

| `P(Healthy)` range | Output label | Rationale |
|---|---|---|
| `>= 0.60` | `Healthy` | Strong positive confidence |
| `<= 0.40` | `Unhealthy` | Strong negative confidence |
| `0.40 - 0.60` | `Borderline` | Uncertain region to reduce overconfident mistakes |

## 8) Interpretation Guide

| Item | Interpretation |
|---|---|
| Positive contribution | Pushes prediction toward `Healthy` |
| Negative contribution | Pushes prediction toward `Unhealthy` |
| Large absolute contribution | Feature is very influential for that specific food |
| Coefficient sign | Dataset-dependent pattern, not universal nutrition truth |
| Engineered feature effect | Encodes domain behavior that raw linear terms miss |

## 9) Operational Checklist

| Check | Requirement |
|---|---|
| Train/inference parity | `app.py` transforms must exactly match `trained_model.py` |
| Feature order parity | Use saved `core_features` / `optional_features` from `trained_model.pkl` |
| Release validation | Evaluate on fixed curated list before deploying |
| Risk tracking | Monitor false negatives (`Unhealthy -> Healthy`) separately |
| Change management | Retrain and regenerate `trained_model.pkl` after feature logic changes |

## 10) Current Model State

| Status | Notes |
|---|---|
| Architecture | Two-model pipeline intact (`core` + `all`) |
| Explainability | Enabled via contribution outputs |
| Robustness | Improved on key historical edge cases through engineered features |
| Usage | Suitable for iterative production-style monitoring and refinement |

## 11) Full Solutions for Current Weaknesses

| Weakness | Root cause | Full solution (implementation) | Success metric |
|---|---|---|---|
| False negatives on high-protein unhealthy foods | Protein can overpower sodium/fat penalties in linear separation | Keep and tune `sodium_protein_risk`, `sat_fat_protein_risk`, `low_fiber_high_fat_protein_risk`; increase curated training examples for processed meats/fast foods | Reduce FN on curated unhealthy set to <=2% |
| Counterintuitive signs for some nutrients | Collinearity and mixed food contexts | Use grouped interaction terms (`refined_carb_density`, `whole_food_fat_protection`) and monitor coefficient stability after each retrain | No major sign flips across 3 consecutive retrains |
| High-confidence wrong predictions | Calibration not explicitly optimized | Add post-training probability calibration (`CalibratedClassifierCV` with sigmoid) and report Brier score + reliability bins | Improve Brier score by >=10% and reduce overconfident wrong predictions |
| Missing optional fields can bias outputs | Imputation assumptions may mismatch real foods | Keep median imputer + `added_sugar` override; add missingness indicator flags per optional field so model learns uncertainty patterns | Accuracy with missing optionals within 2% of full-input accuracy |
| Overfitting risk from many handcrafted rules | Targeted fixes can become narrow | Maintain a frozen holdout + external curated set, and gate releases on both; remove engineered features that do not improve both sets | Holdout and curated metrics both improve or stay stable |
| Limited semantic context (processing method, food type) | Model only sees nutrient numbers per 100g | Add optional metadata model path (food category tags) as separate feature block while preserving nutrient-only default | +1-2% accuracy uplift on edge-case foods |
| Borderline zone still misses some edge cases | Thresholds static across all food patterns | Use threshold tuning by error cost (optimize FN penalty) and consider dynamic confidence warnings for risky profiles | FN rate reduced without >2% drop in total accuracy |

## 12) Implementation Roadmap

| Phase | Scope | Tasks | Output |
|---|---|---|---|
| Phase 1: Reliability hardening | Keep current model family | Add calibration, add missingness indicators, lock evaluation scripts | `trained_model.pkl` vNext + calibration report |
| Phase 2: Data balancing | Improve failure classes | Add more labeled rows for processed meats, refined carbs, and liquid edge cases | Updated dataset + drift report |
| Phase 3: Feature governance | Simplify and stabilize | Keep only features with repeatable gains on holdout + curated tests | Reduced-feature stable model |
| Phase 4: Production monitoring | Continuous quality control | Track FN/FP by category, confidence calibration, and feature drift monthly | Monitoring dashboard + release checklist |

## 13) Acceptance Criteria for Next Release

| Metric | Target |
|---|---|
| Overall curated accuracy | >= 95% |
| False negative rate on unhealthy foods | <= 5% |
| Probability calibration (Brier score) | >= 10% improvement vs current |
| Missing-optionals robustness gap | <= 2% accuracy difference |
| Regression suite pass rate | 100% on critical foods list |

## 14) Coefficient Snapshot (All-features / `models['all']`)

Coefficients below are the raw `coef_[0]` values from the trained logistic regression (standardized features after `StandardScaler`).

Interpretation:
| Coefficient sign | Meaning (on a higher-than-average standardized feature) |
|---|---|
| Positive | Pushes prediction toward `Healthy` |
| Negative | Pushes prediction toward `Unhealthy` |

Note: because features are standardized, the “0” case (feature not triggered) may still produce a non-zero standardized value. Also, coefficients are learned patterns from the training dataset, not universal nutrition causality.

| Feature | Coef (models['all'].coef_[0]) | Role (plain-English) |
|---|---:|---|
| `calories` | -0.5183 | Calorie density context (lower tends to raise Healthy here) |
| `carbohydrates` | +3.0865 | Carb load baseline (dataset learned “more carbs” can align with Healthy) |
| `sugar` | +0.4985 | Sugar baseline (contextual; weaker than other risks) |
| `fat` | -2.6509 | Total fat context (higher tends to reduce Healthy) |
| `saturated_fat` | -6.2624 | Strong fatty/processed risk signal |
| `sodium` | -3.2861 | High sodium correlates with Unhealthy |
| `protein` | +5.3929 | Protein baseline (often aligns with Healthy, but is moderated by risk features) |
| `fried_index_v2` | -3.3242 | Fried/starchy density risk (continuous) |
| `fried_starchy` | -4.4920 | Hard fried-starchy trigger |
| `has_carbs_and_fat` | +1.9224 | Indicates the interaction region is meaningful |
| `net_carbs` | +2.3397 | Refined-vs-fiber-adjusted carbs signal (learned pattern) |
| `sat_fat_protein_risk` | -3.6836 | Penalizes high saturated fat combined with high protein |
| `oil_quality` | +2.4509 | Protects healthy oils (high fat, low sodium, low sat_fat, no added sugar) |
| `sodium_protein_risk` | -6.6538 | Penalizes salty high-protein profiles (triggered when `sodium >= 450`) |
| `refined_carb_density` | -3.1810 | Refined-carb pressure (higher = more likely refined/processed) |
| `whole_food_fat_protection` | +1.4677 | Protects “whole-food” fat sources (higher fiber, low sat_fat) |
| `ultra_refined_carb_penalty` | -0.2673 | Negative-valued refined-carb penalty (coded as `-1` when ultra-refined pattern triggers) |
| `lean_liquid_micronutrient_bonus` | +1.4738 | Preserves healthy low-fat micronutrient-rich profiles |
| `low_fiber_high_fat_protein_risk` | -0.1413 | Penalizes low-fiber high-fat high-protein patterns |
| `processed_refined_carb_salt_risk` | -0.3207 | Refined carbs + salt + low fat + low protein + added sugar risk |
| `fried_protein_density_risk` | -0.1431 | Penalizes “fried-like” dense protein fat profiles |
| `fatty_meat_risk` | -0.7715 | Penalizes fatty meat-like patterns (high protein/fat/sat_fat, low carbs) |
| `refined_carb_sodium_low_fiber_strength` | -0.0445 | Negative-valued refined-carb strength (more salty + low fiber) |
| `refined_carb_density_strong` | +0.4710 | Strong refined-carb density (squared). Because this feature is continuous, its effect is contextual via scaling. |
| `fiber` | +6.5139 | Very strong protective signal in this dataset |
| `cholesterol` | +0.0141 | Weak/near-neutral (often absent → imputed/standardized) |
| `added_sugar` | -4.8889 | Strongest processed-sugar risk (log-transformed + custom imputation) |
| `vitamin_c` | -1.3989 | Vitamin C effect is contextual in this dataset |
| `omega3` | -1.4502 | Omega-3 effect is contextual in this dataset |

## 15) Last-mile Verdict Guard (White Bread)

Even after feature engineering, the UI applies a *rule-based override* to prevent one stubborn false-negative case.

In `app.py` during `/predict`, if the nutrient profile matches:
`carbohydrates > 45` AND `fiber < 3` AND `sodium > 450` AND `saturated_fat < 1`,

then the app clamps probabilities and forces an `Unhealthy` verdict for that request.

This guard is intended only for the refined-carb + low-fiber + salty + low-satfat signature (white-bread-like). It does not change training weights.
