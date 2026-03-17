import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# python -m pip install --upgrade pandas numpy matplotlib scikit-learn 
# run this on terminal

np.random.seed(42)

#declares core or optional features (can add more optional features later, remove this after final edit (the comment only) dev:Joshua)
core_features = [
    "calories", "sugar", "fat", "fiber", "protein",
    "sodium", "cholesterol", "saturated_fat", "carbohydrates"
]
optional_features = [
    "vitamin_c", "calcium", "potassium", "magnesium",
    "iron", "omega3", "monounsaturated_fat", "zinc",
    "phosphorus", "vitamin_a", "vitamin_b6", "vitamin_b12",
    "vitamin_e", "vitamin_k", "choline", "niacin",
]

raw = pd.read_csv('food.csv')
 
column_map = {
    'Data.Carbohydrate'             : 'carbohydrates',
    'Data.Cholesterol'              : 'cholesterol',
    'Data.Fiber'                    : 'fiber',
    'Data.Protein'                  : 'protein',
    'Data.Sugar Total'              : 'sugar',
    'Data.Fat.Polysaturated Fat'    : 'omega3',
    'Data.Fat.Saturated Fat'        : 'saturated_fat',
    'Data.Fat.Total Lipid'          : 'fat',
    'Data.Fat.Monosaturated Fat'    : 'monounsaturated_fat',
    'Data.Major Minerals.Calcium'   : 'calcium',
    'Data.Major Minerals.Iron'      : 'iron',
    'Data.Major Minerals.Magnesium' : 'magnesium',
    'Data.Major Minerals.Potassium' : 'potassium',
    'Data.Major Minerals.Sodium'    : 'sodium',
    'Data.Major Minerals.Zinc'      : 'zinc',
    'Data.Major Minerals.Phosphorus': 'phosphorus',
    'Data.Vitamins.Vitamin C'       : 'vitamin_c',
    'Data.Vitamins.Vitamin A - RAE' : 'vitamin_a',
    'Data.Vitamins.Vitamin B6'      : 'vitamin_b6',
    'Data.Vitamins.Vitamin B12'     : 'vitamin_b12',
    'Data.Vitamins.Vitamin E'       : 'vitamin_e',
    'Data.Vitamins.Vitamin K'       : 'vitamin_k',
    'Data.Choline'                  : 'choline',
    'Data.Niacin'                   : 'niacin',
}
 
df = raw[list(column_map.keys())].copy()
df.rename(columns=column_map, inplace=True)
 
# Calories computed via Atwater formula (not in dataset directly)
df['calories'] = (
    raw['Data.Protein']         * 4 +
    raw['Data.Carbohydrate']    * 4 +
    raw['Data.Fat.Total Lipid'] * 9
)

print(f"Real foods loaded  : {len(df)}")

n_augment   = 10000 - len(df)
augment_idx = np.random.choice(len(df), size=n_augment, replace=True)
augmented   = df.iloc[augment_idx].copy().reset_index(drop=True)
 
for col in df.columns:
    noise_scale = df[col].std() * 0.02   # 2% of std = realistic measurement variation
    augmented[col] = (
        augmented[col] + np.random.normal(0, noise_scale, n_augment)
    ).clip(lower=0)   # nutrition values can't be negative
 
df = pd.concat([df, augmented], ignore_index=True)
print(f"After augmentation : {len(df)} (added {n_augment} rows with 2% noise)")

health_score = (
    # POSITIVE
      df['fiber']               * 4.0    # strongest healthy signal
    + df['protein']             * 1.5    # essential macronutrient
    + df['vitamin_c']           * 0.4    # antioxidant, immune support
    + df['calcium']             * 0.04   # bone health
    + df['potassium']           * 0.012  # heart health, blood pressure
    + df['magnesium']           * 0.15   # metabolic health
    + df['iron']                * 1.5    # oxygen transport
    + df['omega3']              * 6.0    # anti-inflammatory, cardiovascular
    + df['monounsaturated_fat'] * 1.0    # good fat — olive oil, avocado, nuts
    + df['zinc']                * 1.2    # immune function, wound healing
    + df['vitamin_a']           * 0.005  # vision, immune — smaller weight due to large values
    + df['vitamin_e']           * 0.3    # antioxidant, skin health
    + df['vitamin_k']           * 0.01   # blood clotting, bone health
    + df['choline']             * 0.05   # brain health, liver function
    + df['niacin']              * 0.3    # energy metabolism
    + df['vitamin_b6']          * 2.0    # brain health, metabolism
    + df['vitamin_b12']         * 2.0    # nerve function, red blood cells
    + df['phosphorus']          * 0.01   # bone health, energy
    # NEGATIVE
    - df['calories']            * 0.04   # excess energy → weight gain
    - df['carbohydrates']       * 0.15   # refined carbs spike blood sugar
    - df['sugar']               * 2.0    # free sugars
    - df['fat']                 * 0.5    # excess total fat
    - df['sodium']              * 0.015  # raises blood pressure
    - df['cholesterol']         * 0.03   # dietary cholesterol
    - df['saturated_fat']       * 3.5    # raises LDL cholesterol
)

#adding noise to the health score to make it more realistic and less deterministic, since in real life, the healthiness of a food item is not solely determined by its nutrition facts, but also by other factors such as portion size, cooking method, and individual dietary needs. The noise is added to simulate these real-life factors and make the model more robust and generalizable.
THRESHOLD = -15
df['health_label'] = np.where(health_score >= THRESHOLD, 'Healthy', 'Unhealthy')
# 1 = Healthy, 0 = Unhealthy  (matches prediction display in test files)
 
n_healthy   = (df['health_label'] == 'Healthy').sum()
n_unhealthy = (df['health_label'] == 'Unhealthy').sum()
print(f"Labels: Healthy={n_healthy} ({100*n_healthy/len(df):.1f}%)  "
      f"Unhealthy={n_unhealthy} ({100*n_unhealthy/len(df):.1f}%)")

encoder = LabelEncoder()
#convert label into numeric values (unhealthy = 0 and healthy = 1)
df['health_label'] = encoder.fit_transform(df['health_label'])

# LabelEncoder sorts alphabetically: Healthy=0, Unhealthy=1
# Flip so Healthy=1 and Unhealthy=0 (matches prediction display in test files)
df['health_label'] = 1 - df['health_label']

#separating core features and optional features
#only core features
X_core = df[core_features] #no need to use df.drop() since we are only selecting core features only excluding health_label                        

#all features (core + optional)
X_all = df[core_features + optional_features] #no need to use df.drop() since we are selecting all features including core and optional features excluding health_label                    

#target variable (unhealthy = 0 and healthy = 1)
y = df['health_label']

scaler_core = StandardScaler()
scaler_all = StandardScaler()

imputer = SimpleImputer(strategy='mean') # using mean imputation to handle any potential missing values in the dataset, since mean is less sensitive to outliers compared to mean imputation, making it a more robust choice for our dataset which may contain extreme values in nutrition facts.

X_core_scaled = scaler_core.fit_transform(X_core) # scale the core features
X_all_imputed = imputer.fit_transform(X_all) # impute missing values in all features (core + optional)
X_all_scaled = scaler_all.fit_transform(X_all_imputed) # scale all features (core + optional)

#training model using only the core features first

#splitting the data into training and testing sets (80% training and 20% testing) (only core features   and target variable)
X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(
    X_core_scaled, y, test_size=0.2, random_state=42
)

# this line creates a logistic regression model with a maximum of 1000 iterations, this is how we train our model to make predictions based on the data we give it.
# this model is strong if optional features are not filled, but it is weak if optional features are filled, since it relies on core features to make predictions, so if optional features are filled, it will not perform well.                     
model_core = LogisticRegression(max_iter=1000, class_weight='balanced')

#training the model using only 8000(80% of samples) train data (core features only)
model_core.fit(X_train_core, y_train_core)

# making predictions using the test data (core features only)
y_pred_core = model_core.predict(X_test_core)

print("Accuracy using model_core:", accuracy_score(y_test_core, y_pred_core))
print("Confusion Matrix using model_core:\n", confusion_matrix(y_test_core, y_pred_core))
print("Classification Report using model_core:\n", classification_report(y_test_core, y_pred_core))

importance = pd.DataFrame(
    {
        'Healthy': model_core.coef_[0],        # coefficients for class 1
        'Unhealthy': -model_core.coef_[0]      # inverse for class 0
    },
    index=X_core.columns
)
print("Feature importance by class using model_core:\n", importance)

#Training model using all features (core + optional)
#splitting the data into training and testing sets (80% training and 20% testing) (using all features  and target variable)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all_scaled, y, test_size=0.2, random_state=42
)

# this line creates a logistic regression model with a maximum of 1000 iterations, this is how we train our model to make predictions based on the data we give it.
# this model is strong if optional features are filled, but it is weak if optional features are not filled, since it relies on all features to make predictions, so if optional features are not filled, it will not perform well.
model_all = LogisticRegression(max_iter=1000, class_weight='balanced')

#training the model using all features (core + optional)
model_all.fit(X_train_all, y_train_all)

# making predictions using the test data (all features)
y_pred_all = model_all.predict(X_test_all)

print("Accuracy using model_all:", accuracy_score(y_test_all, y_pred_all))
print("Confusion Matrix using model_all:\n", confusion_matrix(y_test_all, y_pred_all))
print("Classification Report using model_all:\n", classification_report(y_test_all, y_pred_all))
importance_all = pd.DataFrame(
    {
        'Healthy': model_all.coef_[0],        # coefficients for class 1
        'Unhealthy': -model_all.coef_[0]      # inverse for class 0
    },
    index=X_all.columns
)

print("Feature importance by class using model_all:\n", importance_all)

trained_models = {
    'core': model_core,
    'all': model_all
}
trained_scalers = {
    'core': scaler_core,   # X_core scaler
    'all': scaler_all     # X_all scaler, or a separate one if you used separate scalers
}

joblib.dump({'models': trained_models, 'scalers': trained_scalers, 'imputer': imputer}, 'trained_model.pkl')