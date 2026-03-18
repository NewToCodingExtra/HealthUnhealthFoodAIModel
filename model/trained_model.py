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
    "calories",
    "carbohydrates",
    "sugar",
    "fat",
    "saturated_fat",
    "sodium",
    "protein",
]
optional_features = [
    "fiber",        # absent/zero in many meat & processed foods
    "cholesterol",  # absent in all plant-based foods
    "added_sugar",  # strongest unhealthy signal — not always on labels
    "vitamin_c",    # strong healthy signal
    "omega3",       # healthy fat signal — distinguishes salmon/nuts from junk
]

df = pd.read_csv('nutrition_data_75k_v2.csv')
 
print(f"Dataset loaded: {len(df)} rows")
 
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

imputer = SimpleImputer(strategy='median') # using mean imputation to handle any potential missing values in the dataset, since mean is less sensitive to outliers compared to mean imputation, making it a more robust choice for our dataset which may contain extreme values in nutrition facts.

X_core_scaled = scaler_core.fit_transform(X_core) # scale the core features
X_all_imputed = imputer.fit_transform(X_all) # impute missing values in all features (core + optional)
X_all_scaled = scaler_all.fit_transform(X_all_imputed) # scale all features (core + optional)

#training model using only the core features first

#splitting the data into training and testing sets (80% training and 20% testing) (only core features   and target variable)
X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(
    X_core_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# this line creates a logistic regression model with a maximum of 1000 iterations, this is how we train our model to make predictions based on the data we give it.
# this model is strong if optional features are not filled, but it is weak if optional features are filled, since it relies on core features to make predictions, so if optional features are filled, it will not perform well.                     
model_core = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0) # C is the inverse of regularization strength, smaller values specify stronger regularization, which can help prevent overfitting when using only core features, especially if some core features are noisy or less relevant. Adjusting C allows us to find a good balance between fitting the training data and generalizing to unseen data.

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
    X_all_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# this line creates a logistic regression model with a maximum of 1000 iterations, this is how we train our model to make predictions based on the data we give it.
# this model is strong if optional features are filled, but it is weak if optional features are not filled, since it relies on all features to make predictions, so if optional features are not filled, it will not perform well.
model_all = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0) # C is the inverse of regularization strength, smaller values specify stronger regularization, which can help prevent overfitting when using all features, especially if some optional features are noisy or less relevant. Adjusting C allows us to find a good balance between fitting the training data and generalizing to unseen data.

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

joblib.dump({'models': trained_models, 'scalers': trained_scalers, 'imputer': imputer, 'optional_features': optional_features, 'core_features': core_features}, 'trained_model.pkl')