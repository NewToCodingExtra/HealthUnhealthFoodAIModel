#TEMPORARY TEST FILE --WILL BE DELETED AFTER UI IS DONE

import joblib
import numpy as np
import pandas as pd

# Load trained models, scalers, and imputer
data = joblib.load('trained_model.pkl')
models = data['models']
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

# Ask user for the food name
food_name = input("Enter the food name: ")

# Collect core features
user_input_core = []
print("\nEnter nutrition facts (core features):")
for feature in core_features:
    value = float(input(f"{feature}: "))
    user_input_core.append(value)

# Collect optional features
# Enter the actual value if known, or press Enter to skip (truly unknown)
# Enter 0 only if the food genuinely has zero of that nutrient
user_input_all = user_input_core.copy()
print("\nEnter optional nutrition facts:")
print("  - Enter the actual value if you know it")
print("  - Enter 0 if the food genuinely has none (e.g. added_sugar in plain salmon)")
print("  - Press Enter (blank) if you don't know — imputer will estimate from similar foods")
for feature in optional_features:
    raw = input(f"  {feature}: ").strip()
    if raw == "":
        # Truly unknown — let imputer estimate
        user_input_all.append(np.nan)
    else:
        user_input_all.append(float(raw))

# Convert to DataFrames for scaling
X_core_df = pd.DataFrame([user_input_core], columns=core_features)
X_all_df  = pd.DataFrame([user_input_all],  columns=core_features + optional_features)

# Scale the input (impute optional unknowns before scaling)
X_core_scaled = scalers['core'].transform(X_core_df)
X_all_imputed = imputer.transform(X_all_df)
X_all_scaled  = scalers['all'].transform(X_all_imputed)

# Make predictions
pred_core = models['core'].predict(X_core_scaled)[0]
pred_all = models['all'].predict(X_all_scaled)[0]

# Display results
print(f"\nFood Item: {food_name}")
print(f"[Core Model] Prediction: {'Healthy' if pred_core else 'Unhealthy'}")
print(f"[All Features Model] Prediction: {'Healthy' if pred_all else 'Unhealthy'}")

# Prediction probabilities
prob_core = models['core'].predict_proba(X_core_scaled)[0]
prob_all = models['all'].predict_proba(X_all_scaled)[0]

print("\nModel Confidence:")
print(f"[Core Model] Healthy Probability: {prob_core[1]:.3f}")
print(f"[Core Model] Unhealthy Probability: {prob_core[0]:.3f}")

print()
print(f"[All Model] Healthy Probability: {prob_all[1]:.3f}")
print(f"[All Model] Unhealthy Probability: {prob_all[0]:.3f}")