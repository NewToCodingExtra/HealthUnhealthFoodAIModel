#TEMPORARY TEST FILE --WILL BE DELETED AFTER UI IS DONE

import joblib
import numpy as np
import pandas as pd

# Load trained models and scalers
data = joblib.load('trained_model.pkl')
models = data['models']
scalers = data['scalers']

core_features = ["calories", "sugar", "fat", "fiber", "protein", "sodium", "cholesterol", "saturated_fat"]
optional_features = ["vitamin_c", "calcium", "added_sugar", "trans_fat"]

# Ask user for the food name
food_name = input("Enter the food name: ")

# Collect core features
user_input_core = []
print("\nEnter nutrition facts (core features):")
for feature in core_features:
    value = float(input(f"{feature}: "))
    user_input_core.append(value)

# Collect optional features
user_input_all = user_input_core.copy()
print("\nEnter optional nutrition facts (enter 0 if unknown):")
for feature in optional_features:
    value = float(input(f"{feature}: "))
    user_input_all.append(value)

# Convert to DataFrames for scaling
X_core_df = pd.DataFrame([user_input_core], columns=core_features)
X_all_df = pd.DataFrame([user_input_all], columns=core_features + optional_features)

# Scale the input
X_core_scaled = scalers['core'].transform(X_core_df)
X_all_scaled = scalers['all'].transform(X_all_df)

# Make predictions
pred_core = models['core'].predict(X_core_scaled)[0]
pred_all = models['all'].predict(X_all_scaled)[0]

# Display results
print(f"\nFood Item: {food_name}")
print(f"[Core Model] Prediction: {'Healthy' if pred_core else 'Unhealthy'}")
print(f"[All Features Model] Prediction: {'Healthy' if pred_all else 'Unhealthy'}")