# TEMPORARY TEST FILE -- WILL BE DELETED AFTER UI IS DONE

import joblib
import pandas as pd

# Load trained models and scalers
data = joblib.load('trained_model.pkl')
models = data['models']
scalers = data['scalers']

core_features = ["calories", "sugar", "fat", "fiber", "protein", "sodium", "cholesterol", "saturated_fat"]
optional_features = ["vitamin_c", "calcium", "added_sugar", "trans_fat"]

# Predefined test foods (core + optional features)
test_foods = [
    {"name": "Apple",          "calories": 52,  "sugar": 10,  "fat": 0.2, "fiber": 2.4,
     "protein": 0.3, "sodium": 1,   "cholesterol": 0,  "saturated_fat": 0.0,
     "vitamin_c": 4.6, "calcium": 6,   "added_sugar": 0,  "trans_fat": 0.0},
    {"name": "Banana",         "calories": 89,  "sugar": 12,  "fat": 0.3, "fiber": 2.6,
     "protein": 1.1, "sodium": 1,   "cholesterol": 0,  "saturated_fat": 0.1,
     "vitamin_c": 8.7, "calcium": 5,   "added_sugar": 0,  "trans_fat": 0.0},
    {"name": "Brown Rice",     "calories": 123, "sugar": 0.3, "fat": 1.0, "fiber": 1.8,
     "protein": 2.6, "sodium": 4,   "cholesterol": 0,  "saturated_fat": 0.2,
     "vitamin_c": 0.0, "calcium": 10,  "added_sugar": 0,  "trans_fat": 0.0},
    {"name": "Orange Juice",   "calories": 45,  "sugar": 9,   "fat": 0.1, "fiber": 0.2,
     "protein": 0.7, "sodium": 1,   "cholesterol": 0,  "saturated_fat": 0.0,
     "vitamin_c": 50, "calcium": 20,  "added_sugar": 0,  "trans_fat": 0.0},
    {"name": "Oatmeal",        "calories": 68,  "sugar": 1,   "fat": 1.4, "fiber": 1.7,
     "protein": 2.4, "sodium": 49,  "cholesterol": 0,  "saturated_fat": 0.2,
     "vitamin_c": 0.0, "calcium": 10,  "added_sugar": 0,  "trans_fat": 0.0},
    {"name": "Grilled Salmon", "calories": 206, "sugar": 0,   "fat": 9.0, "fiber": 0,
     "protein": 30,  "sodium": 59,  "cholesterol": 63, "saturated_fat": 1.4,
     "vitamin_c": 0.0, "calcium": 12,  "added_sugar": 0,  "trans_fat": 0.0},
    {"name": "Lentils",        "calories": 116, "sugar": 1.8, "fat": 0.4, "fiber": 7.9,
     "protein": 9.0, "sodium": 2,   "cholesterol": 0,  "saturated_fat": 0.05,
     "vitamin_c": 1.5, "calcium": 19,  "added_sugar": 0,  "trans_fat": 0.0},
    # Borderline
    {"name": "Whole Milk",     "calories": 60,  "sugar": 5,   "fat": 3.3, "fiber": 0,
     "protein": 3.2, "sodium": 43,  "cholesterol": 10, "saturated_fat": 1.9,
     "vitamin_c": 0.0, "calcium": 120, "added_sugar": 5,  "trans_fat": 0.0},
    {"name": "Whole Egg",      "calories": 155, "sugar": 1.1, "fat": 11,  "fiber": 0,
     "protein": 13,  "sodium": 124, "cholesterol":373, "saturated_fat": 3.3,
     "vitamin_c": 0.0, "calcium": 56,  "added_sugar": 0,  "trans_fat": 0.0},
    # Genuinely unhealthy
    {"name": "Soda",           "calories": 41,  "sugar": 10,  "fat": 0,   "fiber": 0,
     "protein": 0,   "sodium": 1,   "cholesterol": 0,  "saturated_fat": 0.0,
     "vitamin_c": 0.0, "calcium": 0,   "added_sugar": 10, "trans_fat": 0.0},
    {"name": "Chocolate Bar",  "calories": 230, "sugar": 25,  "fat": 13,  "fiber": 2,
     "protein": 3,   "sodium": 50,  "cholesterol": 20, "saturated_fat": 8.0,
     "vitamin_c": 0.0, "calcium": 50,  "added_sugar": 15, "trans_fat": 1.0},
    {"name": "Fried Chicken",  "calories": 246, "sugar": 0,   "fat": 15,  "fiber": 0,
     "protein": 20,  "sodium": 500, "cholesterol": 80, "saturated_fat": 4.0,
     "vitamin_c": 0.0, "calcium": 11,  "added_sugar": 0,  "trans_fat": 2.0},
    {"name": "Cheddar Cheese", "calories": 402, "sugar": 0.5, "fat": 33,  "fiber": 0,
     "protein": 25,  "sodium": 621, "cholesterol":105, "saturated_fat": 21.0,
     "vitamin_c": 0.0, "calcium": 721, "added_sugar": 0,  "trans_fat": 1.0},
    {"name": "French Fries",   "calories": 312, "sugar": 0.3, "fat": 15,  "fiber": 3,
     "protein": 3.4, "sodium": 210, "cholesterol": 0,  "saturated_fat": 2.3,
     "vitamin_c": 7.0, "calcium": 18,  "added_sugar": 0,  "trans_fat": 1.0},
    {"name": "Doughnut",       "calories": 452, "sugar": 27,  "fat": 25,  "fiber": 1.5,
     "protein": 5,   "sodium": 326, "cholesterol": 25, "saturated_fat": 11.0,
     "vitamin_c": 0.0, "calcium": 60,  "added_sugar": 20, "trans_fat": 2.0},
]

# Convert to DataFrame
df_test = pd.DataFrame(test_foods)

# Separate core and all features
X_core = df_test[core_features]
X_all = df_test[core_features + optional_features]

# Scale
X_core_scaled = scalers['core'].transform(X_core)
X_all_scaled = scalers['all'].transform(X_all)

# Make predictions
pred_core = models['core'].predict(X_core_scaled)
pred_all = models['all'].predict(X_all_scaled)

# Display results
for i, row in df_test.iterrows():
    print(f"\nFood Item: {row['name']}")
    print(f"[Core Model] Prediction: {'Healthy' if pred_core[i] else 'Unhealthy'}")
    print(f"[All Features Model] Prediction: {'Healthy' if pred_all[i] else 'Unhealthy'}")