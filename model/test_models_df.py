# TEMPORARY TEST FILE -- WILL BE DELETED AFTER UI IS DONE

import joblib
import pandas as pd
import numpy as np

# Load trained pipelines (imputer + scaler + model bundled together)
data      = joblib.load('trained_model.pkl')
pipelines = data['pipelines']   # {'core': Pipeline, 'all': Pipeline}

# ── FIX (Medium): Unified canonical feature schema ───────────────────────────
# These lists must exactly match trained_model.py and app.py.
# Old test file had fiber/cholesterol in core and carbohydrates as core,
# but the actual training used the schema below. Mismatched schemas gave
# misleading validation results.
core_features = [
    "calories", "carbohydrates", "sugar", "fat",
    "saturated_fat", "sodium", "protein",
]
optional_features = [
    "fiber",        # absent/zero in many meat & processed foods
    "cholesterol",  # absent in all plant-based foods
    "added_sugar",  # strongest unhealthy signal
    "vitamin_c",    # strong healthy signal
    "omega3",       # healthy fat signal
]
all_features = core_features + optional_features

# ─────────────────────────────────────────────────────────────────────────────
# TEST FOODS
# All values per 100 g. Leave optional values as np.nan if genuinely unknown.
# Use 0.0 only when the nutrient is confirmed absent (e.g. added_sugar=0 for
# plain salmon). Do NOT use 0 to mean "I don't know".
# ─────────────────────────────────────────────────────────────────────────────
test_foods = [
    # ── Healthy ──────────────────────────────────────────────────────────────
    {"name": "Apple",
     "calories": 52,  "carbohydrates": 14, "sugar": 10,  "fat": 0.2,
     "saturated_fat": 0.0,  "sodium": 1,   "protein": 0.3,
     "fiber": 2.4,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 4.6,  "omega3": 0.01},

    {"name": "Banana",
     "calories": 89,  "carbohydrates": 23, "sugar": 12,  "fat": 0.3,
     "saturated_fat": 0.1,  "sodium": 1,   "protein": 1.1,
     "fiber": 2.6,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 8.7,  "omega3": 0.03},

    {"name": "Brown Rice",
     "calories": 123, "carbohydrates": 26, "sugar": 0.3, "fat": 1.0,
     "saturated_fat": 0.2,  "sodium": 4,   "protein": 2.6,
     "fiber": 1.8,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 0,    "omega3": 0.03},

    {"name": "Orange Juice",
     "calories": 45,  "carbohydrates": 10, "sugar": 9,   "fat": 0.1,
     "saturated_fat": 0.0,  "sodium": 1,   "protein": 0.7,
     "fiber": 0.2,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 50,   "omega3": 0.01},

    {"name": "Oatmeal",
     "calories": 68,  "carbohydrates": 12, "sugar": 1,   "fat": 1.4,
     "saturated_fat": 0.2,  "sodium": 49,  "protein": 2.4,
     "fiber": 1.7,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 0,    "omega3": 0.04},

    {"name": "Grilled Salmon",
     "calories": 206, "carbohydrates": 0,  "sugar": 0,   "fat": 9.0,
     "saturated_fat": 1.4,  "sodium": 59,  "protein": 30,
     "fiber": 0,    "cholesterol": 63,   "added_sugar": 0,   "vitamin_c": 0,    "omega3": 2.2},

    {"name": "Lentils",
     "calories": 116, "carbohydrates": 20, "sugar": 1.8, "fat": 0.4,
     "saturated_fat": 0.05, "sodium": 2,   "protein": 9.0,
     "fiber": 7.9,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 1.5,  "omega3": 0.09},

    {"name": "Avocado",
     "calories": 160, "carbohydrates": 9,  "sugar": 0.7, "fat": 15,
     "saturated_fat": 2.1,  "sodium": 7,   "protein": 2,
     "fiber": 6.7,  "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 10,   "omega3": 0.1},

    # ── Borderline ────────────────────────────────────────────────────────────
    {"name": "Whole Milk",
     "calories": 60,  "carbohydrates": 5,  "sugar": 5,   "fat": 3.3,
     "saturated_fat": 1.9,  "sodium": 43,  "protein": 3.2,
     "fiber": 0,    "cholesterol": 10,   "added_sugar": 0,   "vitamin_c": 0,    "omega3": 0.08},

    {"name": "Whole Egg",
     "calories": 155, "carbohydrates": 1,  "sugar": 1.1, "fat": 11,
     "saturated_fat": 3.3,  "sodium": 124, "protein": 13,
     "fiber": 0,    "cholesterol": 373,  "added_sugar": 0,   "vitamin_c": 0,    "omega3": 0.1},

    # ── Unhealthy ─────────────────────────────────────────────────────────────
    {"name": "Soda",
     "calories": 41,  "carbohydrates": 11, "sugar": 10,  "fat": 0,
     "saturated_fat": 0.0,  "sodium": 1,   "protein": 0,
     "fiber": 0,    "cholesterol": 0,    "added_sugar": 10,  "vitamin_c": 0,    "omega3": 0},

    {"name": "Chocolate Bar",
     "calories": 230, "carbohydrates": 60, "sugar": 25,  "fat": 13,
     "saturated_fat": 8.0,  "sodium": 50,  "protein": 3,
     "fiber": 2,    "cholesterol": 20,   "added_sugar": 22,  "vitamin_c": 0,    "omega3": 0.05},

    {"name": "Fried Chicken",
     "calories": 246, "carbohydrates": 8,  "sugar": 0,   "fat": 15,
     "saturated_fat": 4.0,  "sodium": 600, "protein": 20,
     "fiber": 0,    "cholesterol": 80,   "added_sugar": 0,   "vitamin_c": 0,    "omega3": 0.1},

    {"name": "Cheddar Cheese",
     "calories": 402, "carbohydrates": 1,  "sugar": 0.5, "fat": 33,
     "saturated_fat": 21.0, "sodium": 621, "protein": 25,
     "fiber": 0,    "cholesterol": 105,  "added_sugar": 0,   "vitamin_c": 0,    "omega3": 0.4},

    {"name": "French Fries",
     "calories": 312, "carbohydrates": 41, "sugar": 0.3, "fat": 15,
     "saturated_fat": 2.3,  "sodium": 210, "protein": 3.4,
     "fiber": 3,    "cholesterol": 0,    "added_sugar": 0,   "vitamin_c": 7,    "omega3": 0.05},

    {"name": "Doughnut",
     "calories": 452, "carbohydrates": 51, "sugar": 27,  "fat": 25,
     "saturated_fat": 11.0, "sodium": 326, "protein": 5,
     "fiber": 1.5,  "cholesterol": 25,   "added_sugar": 18,  "vitamin_c": 0,    "omega3": 0.1},
]

df_test = pd.DataFrame(test_foods)

X_core = df_test[core_features]
X_all  = df_test[all_features]

# Pipelines handle imputation and scaling internally — no manual transform needed
prob_core = pipelines['core'].predict_proba(X_core)[:, 1]
prob_all  = pipelines['all'].predict_proba(X_all)[:, 1]
pred_core = pipelines['core'].predict(X_core)
pred_all  = pipelines['all'].predict(X_all)

for i, row in df_test.iterrows():
    core_label = 'Healthy  ' if pred_core[i] else 'Unhealthy'
    all_label  = 'Healthy  ' if pred_all[i]  else 'Unhealthy'
    warning    = '  ⚠️  Models disagree — try entering optional features' \
                 if pred_core[i] != pred_all[i] else ''
    print(f"\nFood Item: {row['name']}{warning}")
    print(f"[Core Model]         {core_label} ({prob_core[i]*100:.1f}% confident)")
    print(f"[All Features Model] {all_label}  ({prob_all[i]*100:.1f}% confident)")