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
    "vitamin_c", "calcium", "potassium", "magnesium",
    "iron", "omega3", "monounsaturated_fat", "zinc",
    "phosphorus", "vitamin_a", "vitamin_b6", "vitamin_b12",
    "vitamin_e", "vitamin_k", "choline", "niacin", "added_sugar",
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
     "vitamin_c": 4.6,"calcium": 6,   "potassium": 107, "magnesium": 5,
     "iron": 0.1,     "omega3": 0.01, "monounsaturated_fat": 0.01, "zinc": 0.04,
     "phosphorus": 11,"vitamin_a": 3,  "vitamin_b6": 0.04, "vitamin_b12": 0,
     "vitamin_e": 0.2,"vitamin_k": 2.2,"choline": 3.4,    "niacin": 0.09,  "added_sugar": 0},

    {"name": "Banana",
     "calories": 89,  "sugar": 12,  "fat": 0.3,  "fiber": 2.6,  "protein": 1.1,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.1,  "carbohydrates": 23,
     "vitamin_c": 8.7,"calcium": 5,   "potassium": 358, "magnesium": 27,
     "iron": 0.3,     "omega3": 0.03, "monounsaturated_fat": 0.03, "zinc": 0.15,
     "phosphorus": 22,"vitamin_a": 3,  "vitamin_b6": 0.37, "vitamin_b12": 0,
     "vitamin_e": 0.1,"vitamin_k": 0.5,"choline": 9.8,    "niacin": 0.67,  "added_sugar": 0},

    {"name": "Brown Rice",
     "calories": 123, "sugar": 0.3, "fat": 1.0,  "fiber": 1.8,  "protein": 2.6,
     "sodium": 4,     "cholesterol": 0,   "saturated_fat": 0.2,  "carbohydrates": 26,
     "vitamin_c": 0,  "calcium": 10,  "potassium": 43,  "magnesium": 44,
     "iron": 0.5,     "omega3": 0.03, "monounsaturated_fat": 0.3,  "zinc": 0.6,
     "phosphorus": 83,"vitamin_a": 0,  "vitamin_b6": 0.15, "vitamin_b12": 0,
     "vitamin_e": 0.1,"vitamin_k": 1.9,"choline": 9.2,    "niacin": 1.53,  "added_sugar": 0},

    {"name": "Orange Juice",
     "calories": 45,  "sugar": 9,   "fat": 0.1,  "fiber": 0.2,  "protein": 0.7,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.0,  "carbohydrates": 10,
     "vitamin_c": 50, "calcium": 20,  "potassium": 200, "magnesium": 11,
     "iron": 0.2,     "omega3": 0.01, "monounsaturated_fat": 0.02, "zinc": 0.05,
     "phosphorus": 17,"vitamin_a": 10, "vitamin_b6": 0.04, "vitamin_b12": 0,
     "vitamin_e": 0.1,"vitamin_k": 0.1,"choline": 8.4,    "niacin": 0.4,  "added_sugar": 0},

    {"name": "Oatmeal",
     "calories": 68,  "sugar": 1,   "fat": 1.4,  "fiber": 1.7,  "protein": 2.4,
     "sodium": 49,    "cholesterol": 0,   "saturated_fat": 0.2,  "carbohydrates": 12,
     "vitamin_c": 0,  "calcium": 10,  "potassium": 61,  "magnesium": 26,
     "iron": 0.7,     "omega3": 0.04, "monounsaturated_fat": 0.4,  "zinc": 0.6,
     "phosphorus": 77,"vitamin_a": 0,  "vitamin_b6": 0.02, "vitamin_b12": 0,
     "vitamin_e": 0.1,"vitamin_k": 0.5,"choline": 7.0,    "niacin": 0.15,  "added_sugar": 0},

    {"name": "Grilled Salmon",
     "calories": 206, "sugar": 0,   "fat": 9.0,  "fiber": 0,    "protein": 30,
     "sodium": 59,    "cholesterol": 63,  "saturated_fat": 1.4,  "carbohydrates": 0,
     "vitamin_c": 0,  "calcium": 12,  "potassium": 490, "magnesium": 29,
     "iron": 0.3,     "omega3": 2.2,  "monounsaturated_fat": 3.0,  "zinc": 0.4,
     "phosphorus": 371,"vitamin_a": 12, "vitamin_b6": 0.9,  "vitamin_b12": 3.2,
     "vitamin_e": 1.1,"vitamin_k": 0.5,"choline": 96.0,   "niacin": 8.6,  "added_sugar": 0},

    {"name": "Lentils",
     "calories": 116, "sugar": 1.8, "fat": 0.4,  "fiber": 7.9,  "protein": 9.0,
     "sodium": 2,     "cholesterol": 0,   "saturated_fat": 0.05, "carbohydrates": 20,
     "vitamin_c": 1.5,"calcium": 19,  "potassium": 369, "magnesium": 36,
     "iron": 3.3,     "omega3": 0.09, "monounsaturated_fat": 0.07, "zinc": 1.3,
     "phosphorus": 180,"vitamin_a": 1,  "vitamin_b6": 0.18, "vitamin_b12": 0,
     "vitamin_e": 0.1,"vitamin_k": 1.7,"choline": 32.5,   "niacin": 1.06,  "added_sugar": 0},

    # ── Borderline ────────────────────────────────────────────────────────────
    {"name": "Whole Milk",
     "calories": 60,  "sugar": 5,   "fat": 3.3,  "fiber": 0,    "protein": 3.2,
     "sodium": 43,    "cholesterol": 10,  "saturated_fat": 1.9,  "carbohydrates": 5,
     "vitamin_c": 0,  "calcium": 120, "potassium": 150, "magnesium": 10,
     "iron": 0.1,     "omega3": 0.08, "monounsaturated_fat": 0.8,  "zinc": 0.4,
     "phosphorus": 93,"vitamin_a": 46, "vitamin_b6": 0.04, "vitamin_b12": 0.45,
     "vitamin_e": 0.1,"vitamin_k": 0.3,"choline": 14.3,   "niacin": 0.09,  "added_sugar": 5},

    {"name": "Whole Egg",
     "calories": 155, "sugar": 1.1, "fat": 11,   "fiber": 0,    "protein": 13,
     "sodium": 124,   "cholesterol": 373, "saturated_fat": 3.3,  "carbohydrates": 1,
     "vitamin_c": 0,  "calcium": 56,  "potassium": 138, "magnesium": 12,
     "iron": 1.8,     "omega3": 0.1,  "monounsaturated_fat": 4.1,  "zinc": 1.3,
     "phosphorus": 198,"vitamin_a": 149,"vitamin_b6": 0.17, "vitamin_b12": 0.89,
     "vitamin_e": 1.1,"vitamin_k": 0.3,"choline": 294.0,  "niacin": 0.07,  "added_sugar": 0},

    # ── Unhealthy ─────────────────────────────────────────────────────────────
    {"name": "Soda",
     "calories": 41,  "sugar": 10,  "fat": 0,    "fiber": 0,    "protein": 0,
     "sodium": 1,     "cholesterol": 0,   "saturated_fat": 0.0,  "carbohydrates": 11,
     "vitamin_c": 0,  "calcium": 0,   "potassium": 2,   "magnesium": 0,
     "iron": 0,       "omega3": 0,    "monounsaturated_fat": 0,    "zinc": 0,
     "phosphorus": 0, "vitamin_a": 0,  "vitamin_b6": 0,    "vitamin_b12": 0,
     "vitamin_e": 0,  "vitamin_k": 0,  "choline": 0,      "niacin": 0,  "added_sugar": 10},

    {"name": "Chocolate Bar",
     "calories": 230, "sugar": 25,  "fat": 13,   "fiber": 2,    "protein": 3,
     "sodium": 50,    "cholesterol": 20,  "saturated_fat": 8.0,  "carbohydrates": 60,
     "vitamin_c": 0,  "calcium": 50,  "potassium": 200, "magnesium": 40,
     "iron": 2.0,     "omega3": 0.05, "monounsaturated_fat": 4.5,  "zinc": 0.9,
     "phosphorus": 87,"vitamin_a": 0,  "vitamin_b6": 0.05, "vitamin_b12": 0.2,
     "vitamin_e": 0.5,"vitamin_k": 4.8,"choline": 20.0,   "niacin": 0.4,  "added_sugar": 22},

    {"name": "Fried Chicken",
     "calories": 246, "sugar": 0,   "fat": 15,   "fiber": 0,    "protein": 20,
     "sodium": 600,   "cholesterol": 80,  "saturated_fat": 4.0,  "carbohydrates": 18,
     "vitamin_c": 0,  "calcium": 11,  "potassium": 220, "magnesium": 20,
     "iron": 1.0,     "omega3": 0.1,  "monounsaturated_fat": 6.0,  "zinc": 1.5,
     "phosphorus": 156,"vitamin_a": 21, "vitamin_b6": 0.4,  "vitamin_b12": 0.3,
     "vitamin_e": 0.5,"vitamin_k": 4.3,"choline": 60.0,   "niacin": 6.8,  "added_sugar": 0},

    {"name": "Cheddar Cheese",
     "calories": 402, "sugar": 0.5, "fat": 33,   "fiber": 0,    "protein": 25,
     "sodium": 621,   "cholesterol": 105, "saturated_fat": 21.0, "carbohydrates": 1,
     "vitamin_c": 0,  "calcium": 721, "potassium": 98,  "magnesium": 28,
     "iron": 0.7,     "omega3": 0.4,  "monounsaturated_fat": 9.4,  "zinc": 3.1,
     "phosphorus": 512,"vitamin_a": 264,"vitamin_b6": 0.07, "vitamin_b12": 0.83,
     "vitamin_e": 0.3,"vitamin_k": 2.8,"choline": 15.4,   "niacin": 0.06,  "added_sugar": 0},

    {"name": "French Fries",
     "calories": 312, "sugar": 0.3, "fat": 15,   "fiber": 3,    "protein": 3.4,
     "sodium": 210,   "cholesterol": 0,   "saturated_fat": 2.3,  "carbohydrates": 41,
     "vitamin_c": 7,  "calcium": 18,  "potassium": 535, "magnesium": 30,
     "iron": 1.0,     "omega3": 0.05, "monounsaturated_fat": 6.5,  "zinc": 0.4,
     "phosphorus": 85,"vitamin_a": 0,  "vitamin_b6": 0.3,  "vitamin_b12": 0,
     "vitamin_e": 1.5,"vitamin_k": 8.4,"choline": 14.0,   "niacin": 2.3,  "added_sugar": 0},

    {"name": "Doughnut",
     "calories": 452, "sugar": 27,  "fat": 25,   "fiber": 1.5,  "protein": 5,
     "sodium": 326,   "cholesterol": 25,  "saturated_fat": 11.0, "carbohydrates": 51,
     "vitamin_c": 0,  "calcium": 60,  "potassium": 90,  "magnesium": 15,
     "iron": 1.2,     "omega3": 0.1,  "monounsaturated_fat": 11.0, "zinc": 0.4,
     "phosphorus": 72,"vitamin_a": 0,  "vitamin_b6": 0.04, "vitamin_b12": 0.1,
     "vitamin_e": 2.1,"vitamin_k": 6.4,"choline": 18.0,   "niacin": 1.9,  "added_sugar": 18},
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