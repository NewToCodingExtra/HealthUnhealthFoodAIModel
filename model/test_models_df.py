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
     {"food_name": "Whole Milk", "calories": 60, "sugar": 5, "fat": 3.3, "fiber": 0,
     "protein": 3.2, "sodium": 43, "cholesterol": 10, "saturated_fat": 1.9,
     "vitamin_c": 0, "calcium": 120, "added_sugar": 5, "trans_fat": 0},
    
    {"food_name": "Soda", "calories": 41, "sugar": 10, "fat": 0, "fiber": 0,
     "protein": 0, "sodium": 1, "cholesterol": 0, "saturated_fat": 0,
     "vitamin_c": 0, "calcium": 0, "added_sugar": 10, "trans_fat": 0},
    
    {"food_name": "Apple", "calories": 52, "sugar": 10, "fat": 0.2, "fiber": 2.4,
     "protein": 0.3, "sodium": 1, "cholesterol": 0, "saturated_fat": 0,
     "vitamin_c": 4.6, "calcium": 6, "added_sugar": 0, "trans_fat": 0},
    
    {"food_name": "Chocolate Bar", "calories": 230, "sugar": 25, "fat": 13, "fiber": 2,
     "protein": 3, "sodium": 50, "cholesterol": 20, "saturated_fat": 8,
     "vitamin_c": 0, "calcium": 50, "added_sugar": 15, "trans_fat": 1},
    
    {"food_name": "Banana", "calories": 89, "sugar": 12, "fat": 0.3, "fiber": 2.6,
     "protein": 1.1, "sodium": 1, "cholesterol": 0, "saturated_fat": 0.1,
     "vitamin_c": 8.7, "calcium": 5, "added_sugar": 0, "trans_fat": 0},
    
    {"food_name": "Cheddar Cheese", "calories": 402, "sugar": 0.5, "fat": 33, "fiber": 0,
     "protein": 25, "sodium": 621, "cholesterol": 105, "saturated_fat": 21,
     "vitamin_c": 0, "calcium": 721, "added_sugar": 0, "trans_fat": 1},
    
    {"food_name": "Orange Juice", "calories": 45, "sugar": 9, "fat": 0.1, "fiber": 0.2,
     "protein": 0.7, "sodium": 1, "cholesterol": 0, "saturated_fat": 0,
     "vitamin_c": 50, "calcium": 20, "added_sugar": 0, "trans_fat": 0},
    
    {"food_name": "Fried Chicken", "calories": 246, "sugar": 0, "fat": 15, "fiber": 0,
     "protein": 20, "sodium": 500, "cholesterol": 80, "saturated_fat": 4,
     "vitamin_c": 0, "calcium": 11, "added_sugar": 0, "trans_fat": 2},
    
    {"food_name": "Brown Rice", "calories": 123, "sugar": 0.3, "fat": 1, "fiber": 1.8,
     "protein": 2.6, "sodium": 4, "cholesterol": 0, "saturated_fat": 0.2,
     "vitamin_c": 0, "calcium": 10, "added_sugar": 0, "trans_fat": 0},
    
    {"food_name": "French Fries", "calories": 312, "sugar": 0.3, "fat": 15, "fiber": 3,
     "protein": 3.4, "sodium": 210, "cholesterol": 0, "saturated_fat": 2.3,
     "vitamin_c": 7, "calcium": 18, "added_sugar": 0, "trans_fat": 1}
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
    print(f"\nFood Item: {row['food_name']}")
    print(f"[Core Model] Prediction: {'Healthy' if pred_core[i] else 'Unhealthy'}")
    print(f"[All Features Model] Prediction: {'Healthy' if pred_all[i] else 'Unhealthy'}")