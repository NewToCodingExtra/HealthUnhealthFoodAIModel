# TEMPORARY TEST FILE -- WILL BE DELETED AFTER UI IS DONE

import joblib
import pandas as pd
import numpy as np

# Load trained models, scalers, and imputer
data    = joblib.load('trained_model.pkl')
models  = data['models']
scalers = data['scalers']
imputer = data['imputer']

core_features = [
    "calories", "sugar", "fat", "fiber", "protein",
    "sodium", "cholesterol", "saturated_fat", "carbohydrates"
]
optional_features = [
    "added_sugar",
    "vitamin_c",
    "omega3",
]

# ─────────────────────────────────────────────────────────────────────────────
# TEST FOODS
# All values per 100g. Enter 0 for any unknown optional value.
# New core field : carbohydrates (g)
# New optional fields: potassium, magnesium, iron, omega3,
#                      monounsaturated_fat, zinc, phosphorus,
#                      vitamin_a, vitamin_b6, vitamin_b12,
#                      vitamin_e, vitamin_k, choline, niacin
# Removed: added_sugar, trans_fat (not in training dataset)
# ─────────────────────────────────────────────────────────────────────────────
test_foods = [
    # ── Healthy ──────────────────────────────────────────────────────────────
    {"name": "Apple",
     "calories": 52,  "sugar": 10,  "fat": 0.2,  "fiber": 2.4,  "protein": 0.3,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.0,  "carbohydrates": 14,
     "added_sugar": 0,    "vitamin_c": 4.6,  "omega3": 0.01},

    {"name": "Banana",
     "calories": 89,  "sugar": 12,  "fat": 0.3,  "fiber": 2.6,  "protein": 1.1,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.1,  "carbohydrates": 23,
     "added_sugar": 0,    "vitamin_c": 8.7,  "omega3": 0.03},

    {"name": "Brown Rice",
     "calories": 123, "sugar": 0.3, "fat": 1.0,  "fiber": 1.8,  "protein": 2.6,
     "sodium": 4,     "cholesterol": 0,   "saturated_fat": 0.2,  "carbohydrates": 26,
     "added_sugar": 0,    "vitamin_c": 0,    "omega3": 0.03},

    {"name": "Orange Juice",
     "calories": 45,  "sugar": 9,   "fat": 0.1,  "fiber": 0.2,  "protein": 0.7,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.0,  "carbohydrates": 10,
     "added_sugar": 0,    "vitamin_c": 50,   "omega3": 0.01},

    {"name": "Oatmeal",
     "calories": 68,  "sugar": 1,   "fat": 1.4,  "fiber": 1.7,  "protein": 2.4,
     "sodium": 49,    "cholesterol": 0,   "saturated_fat": 0.2,  "carbohydrates": 12,
     "added_sugar": 0,    "vitamin_c": 0,    "omega3": 0.04},

    {"name": "Grilled Salmon",
     "calories": 206, "sugar": 0,   "fat": 9.0,  "fiber": 0,    "protein": 30,
     "sodium": 59,    "cholesterol": 63,  "saturated_fat": 1.4,  "carbohydrates": 0,
     "added_sugar": 0,    "vitamin_c": 0,    "omega3": 2.2},

    {"name": "Lentils",
     "calories": 116, "sugar": 1.8, "fat": 0.4,  "fiber": 7.9,  "protein": 9.0,
     "sodium": 2,     "cholesterol": 0,   "saturated_fat": 0.05, "carbohydrates": 20,
     "added_sugar": 0,    "vitamin_c": 1.5,  "omega3": 0.09},

    {"name": "Avocado",
     "calories": 160, "sugar": 0.7, "fat": 15,   "fiber": 6.7,  "protein": 2,
     "sodium": 7,     "cholesterol": 0,   "saturated_fat": 2.1,  "carbohydrates": 9,
     "added_sugar": 0,    "vitamin_c": 10,   "omega3": 0.1},

    # ── Borderline ────────────────────────────────────────────────────────────
    {"name": "Whole Milk",
     "calories": 60,  "sugar": 5,   "fat": 3.3,  "fiber": 0,    "protein": 3.2,
     "sodium": 43,    "cholesterol": 10,  "saturated_fat": 1.9,  "carbohydrates": 5,
     "added_sugar": 5,    "vitamin_c": 0,    "omega3": 0.08},

    {"name": "Whole Egg",
     "calories": 155, "sugar": 1.1, "fat": 11,   "fiber": 0,    "protein": 13,
     "sodium": 124,   "cholesterol": 373, "saturated_fat": 3.3,  "carbohydrates": 1,
     "added_sugar": 0,    "vitamin_c": 0,    "omega3": 0.1},

    # ── Unhealthy ─────────────────────────────────────────────────────────────
    {"name": "Soda",
     "calories": 41,  "sugar": 10,  "fat": 0,    "fiber": 0,    "protein": 0,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.0,  "carbohydrates": 11,
     "added_sugar": 10,   "vitamin_c": 0,    "omega3": 0},

    {"name": "Chocolate Bar",
     "calories": 230, "sugar": 25,  "fat": 13,   "fiber": 2,    "protein": 3,
     "sodium": 50,    "cholesterol": 20,  "saturated_fat": 8.0,  "carbohydrates": 60,
     "added_sugar": 22,   "vitamin_c": 0,    "omega3": 0.05},

    {"name": "Fried Chicken",
     "calories": 246, "sugar": 0,   "fat": 15,   "fiber": 0,    "protein": 20,
     "sodium": 600,   "cholesterol": 80,  "saturated_fat": 4.0,  "carbohydrates": 8,
     "added_sugar": 0,    "vitamin_c": 0,    "omega3": 0.1},

    {"name": "Cheddar Cheese",
     "calories": 402, "sugar": 0.5, "fat": 33,   "fiber": 0,    "protein": 25,
     "sodium": 621,   "cholesterol": 105, "saturated_fat": 21.0, "carbohydrates": 1,
     "added_sugar": 0,    "vitamin_c": 0,    "omega3": 0.4},

    {"name": "French Fries",
     "calories": 312, "sugar": 0.3, "fat": 15,   "fiber": 3,    "protein": 3.4,
     "sodium": 210,   "cholesterol": 0,   "saturated_fat": 2.3,  "carbohydrates": 41,
     "added_sugar": 0,    "vitamin_c": 7,    "omega3": 0.05},

    {"name": "Doughnut",
     "calories": 452, "sugar": 27,  "fat": 25,   "fiber": 1.5,  "protein": 5,
     "sodium": 326,   "cholesterol": 25,  "saturated_fat": 11.0, "carbohydrates": 51,
     "added_sugar": 18,   "vitamin_c": 0,    "omega3": 0.1},
]

df_test = pd.DataFrame(test_foods)

X_core = df_test[core_features]
X_all  = df_test[core_features + optional_features]

X_core_scaled = scalers['core'].transform(X_core)
X_all_imputed = imputer.transform(X_all)
X_all_scaled  = scalers['all'].transform(X_all_imputed)

pred_core = models['core'].predict(X_core_scaled)
pred_all  = models['all'].predict(X_all_scaled)
prob_core = models['core'].predict_proba(X_core_scaled)[:, 1]
prob_all  = models['all'].predict_proba(X_all_scaled)[:, 1]

for i, row in df_test.iterrows():
    core_label = 'Healthy  ' if pred_core[i] else 'Unhealthy'
    all_label  = 'Healthy  ' if pred_all[i]  else 'Unhealthy'
    warning    = '  ⚠️  Models disagree — try entering optional features' \
                 if pred_core[i] != pred_all[i] else ''
    print(f"\nFood Item: {row['name']}{warning}")
    print(f"[Core Model]         {core_label} ({prob_core[i]*100:.1f}% confident)")
    print(f"[All Features Model] {all_label}  ({prob_all[i]*100:.1f}% confident)")