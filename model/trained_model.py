import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# python -m pip install --upgrade pandas numpy matplotlib scikit-learn 
# run this on terminal

np.random.seed(42)

# declares core or optional features
core_features = [
    "calories",
    "carbohydrates",
    "sugar",
    "fat",
    "saturated_fat",
    "sodium",
    "protein",
    # derivatives features
    "fried_index_v2",
    "fried_starchy",
    "has_carbs_and_fat",
    "net_carbs",
    "sat_fat_protein_risk",
    "oil_quality",
    "sodium_protein_risk",
    "refined_carb_density",
    "whole_food_fat_protection",
    "ultra_refined_carb_penalty",
    "lean_liquid_micronutrient_bonus",
    "low_fiber_high_fat_protein_risk",
    "processed_refined_carb_salt_risk",
    "fried_protein_density_risk",
    "fatty_meat_risk",
    "refined_carb_sodium_low_fiber_strength",
    "refined_carb_density_strong",
]
optional_features = [
    "fiber",
    "cholesterol",
    "added_sugar",
    "vitamin_c",
    "omega3",
]

df = pd.read_csv('nutrition_data_125k.csv')

# Log-transform added_sugar first.
df['added_sugar'] = np.log1p(df['added_sugar'])
# Stabilize optional nutrient scales with heavy tails.
df['omega3'] = np.log1p(np.clip(df['omega3'], 0, 2.0))
df['vitamin_c'] = np.log1p(np.clip(df['vitamin_c'], 0, 120.0))

# Engineered features.
df['fried_index_v2'] = np.where(
    (df['carbohydrates'] >= 10) & (df['fat'] >= 5),
    np.log1p((df['carbohydrates'] * df['fat']) / 100),
    0.0
)
df['fried_starchy'] = ((df['carbohydrates'] > 35) & (df['fat'] > 8)).astype(float)
df['has_carbs_and_fat'] = ((df['carbohydrates'] > 2) & (df['fat'] > 2)).astype(float)
df['net_carbs'] = np.maximum(df['carbohydrates'] - df['fiber'].fillna(0), 0)
df['sat_fat_protein_risk'] = np.where(
    df['saturated_fat'] > 6,
    df['saturated_fat'] * (df['protein'] / 30.0),
    0.0
)
df['oil_quality'] = (
    (df['fat'] > 70)
    & (df['sodium'] < 10)
    & (df['saturated_fat'] < 20)
    & (df['added_sugar'] == 0)
).astype(float)
df['sodium_protein_risk'] = np.where(
    df['sodium'] >= 450,
    (df['sodium'] / 300.0) * (1.0 + (df['protein'] / 30.0)),
    0.0
)
df['refined_carb_density'] = df['carbohydrates'] / (df['fiber'].fillna(0.1) + 1.0)
df['whole_food_fat_protection'] = (
    (df['fat'] > 35)
    & (df['fiber'].fillna(0) > 4)
    & (df['saturated_fat'] < 12)
    & (df['added_sugar'] <= np.log1p(6))
).astype(float)
df['ultra_refined_carb_penalty'] = (
    (df['carbohydrates'] > 40)
    & (df['fiber'].fillna(0) < 3.5)
    & (df['sodium'] > 200)
).astype(float) * -1.0
df['lean_liquid_micronutrient_bonus'] = (
    (df['vitamin_c'] > np.log1p(20))
    & (df['sodium'] < 20)
    & (df['fat'] < 1)
    & (df['saturated_fat'] < 1)
    & (df['protein'] < 2)
).astype(float)
df['low_fiber_high_fat_protein_risk'] = (
    (df['protein'] > 18)
    & (df['fat'] > 12)
    & (df['fiber'].fillna(0) < 1)
).astype(float)
df['processed_refined_carb_salt_risk'] = (
    (df['carbohydrates'] > 40)
    & (df['sodium'] > 350)
    & (df['fat'] < 6)
    & (df['protein'] < 11)
    & (df['added_sugar'] > np.log1p(1))
).astype(float)
df['fried_protein_density_risk'] = np.where(
    (df['protein'] > 18) & (df['fat'] > 12) & (df['sodium'] > 450),
    (df['protein'] * df['fat']) / (df['fiber'].fillna(0) + 1.0),
    0.0
)
df['fatty_meat_risk'] = (
    (df['protein'] > 20)
    & (df['fat'] > 15)
    & (df['saturated_fat'] > 5)
    & (df['carbohydrates'] < 5)
).astype(float)
df['refined_carb_sodium_low_fiber_strength'] = np.where(
    (df['carbohydrates'] > 45)
    & (df['fiber'].fillna(0) < 3)
    & (df['sodium'] > 300),
    -1.0 * (df['carbohydrates'] / (df['fiber'].fillna(0) + 1.0)) * (df['sodium'] / 500.0),
    0.0
)
df['refined_carb_density_strong'] = (
    (df['carbohydrates'] / (df['fiber'].fillna(0.1) + 1.0)) ** 2
)

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

imputer = SimpleImputer(strategy='median')

X_core_scaled = scaler_core.fit_transform(X_core) # scale the core features
imputer.fit(X_all)
added_sugar_idx = (core_features + optional_features).index('added_sugar')
imputer.statistics_[added_sugar_idx] = np.log1p(0.5)
X_all_imputed = imputer.transform(X_all) # impute missing values in all features (core + optional)
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

# ── Calibration (reliability curve + Brier score) ─────────────────────────────
# We use the same held-out splits from train_test_split above, so calibration is
# measured on unseen data (not training).
n_bins = 10

proba_core_healthy = model_core.predict_proba(X_test_core)[:, 1]  # P(Healthy)
proba_all_healthy  = model_all.predict_proba(X_test_all)[:, 1]   # P(Healthy)

brier_core = float(brier_score_loss(y_test_core, proba_core_healthy))
brier_all  = float(brier_score_loss(y_test_all, proba_all_healthy))

core_prob_true, core_prob_pred = calibration_curve(
    y_test_core, proba_core_healthy, n_bins=n_bins, strategy='uniform'
)
all_prob_true, all_prob_pred = calibration_curve(
    y_test_all, proba_all_healthy, n_bins=n_bins, strategy='uniform'
)

calibration_payload = {
    'core': {
        'brier': brier_core,
        'reliability': {
            'prob_pred': core_prob_pred.tolist(),
            'prob_true': core_prob_true.tolist(),
        }
    },
    'all': {
        'brier': brier_all,
        'reliability': {
            'prob_pred': all_prob_pred.tolist(),
            'prob_true': all_prob_true.tolist(),
        }
    }
}

trained_models = {
    'core': model_core,
    'all': model_all
}
trained_scalers = {
    'core': scaler_core,   # X_core scaler
    'all': scaler_all     # X_all scaler, or a separate one if you used separate scalers
}

joblib.dump({
    'models':                    trained_models,
    'scalers':                   trained_scalers,
    'imputer':                   imputer,
    'optional_features':         optional_features,
    'core_features':             core_features,
    'log_transformed_features':  ['added_sugar', 'omega3', 'vitamin_c'],
    'calibration':               calibration_payload,
    'engineered_features': {
        'fried_index_v2': 'if carbohydrates >= 10 and fat >= 5 then log1p(carbohydrates * fat / 100) else 0',
        'fried_starchy': '1 if carbohydrates > 35 and fat > 8 else 0',
        'has_carbs_and_fat': '1 if carbohydrates > 2 and fat > 2 else 0',
        'net_carbs': 'max(carbohydrates - fiber, 0)',
        'sat_fat_protein_risk': 'if saturated_fat > 6 then saturated_fat * (protein / 30) else 0',
        'oil_quality': '1 if fat > 70 and sodium < 10 and saturated_fat < 20 and added_sugar == 0 else 0',
        'sodium_protein_risk': 'if sodium >= 450 then (sodium / 300) * (1 + protein / 30) else 0',
        'refined_carb_density': 'carbohydrates / (fiber + 1), with missing fiber filled to 0.1',
        'whole_food_fat_protection': '1 if fat > 35 and fiber > 4 and saturated_fat < 12 and added_sugar <= log1p(6) else 0',
        'ultra_refined_carb_penalty': '1 if carbohydrates > 40 and fiber < 3.5 and sodium > 200 else 0',
        'lean_liquid_micronutrient_bonus': '1 if vitamin_c > log1p(20) and sodium < 20 and fat < 1 and saturated_fat < 1 and protein < 2 else 0',
        'low_fiber_high_fat_protein_risk': '1 if protein > 18 and fat > 12 and fiber < 1 else 0',
        'processed_refined_carb_salt_risk': '1 if carbohydrates > 40 and sodium > 350 and fat < 6 and protein < 11 and added_sugar > log1p(1) else 0',
        'fried_protein_density_risk': 'if protein > 18 and fat > 12 and sodium > 450 then (protein * fat) / (fiber + 1) else 0',
        'fatty_meat_risk': '1 if protein > 20 and fat > 15 and saturated_fat > 5 and carbohydrates < 5 else 0',
        'refined_carb_sodium_low_fiber_strength': 'if carbohydrates > 45 and fiber < 3 and sodium > 300 then (carbohydrates / (fiber + 1)) * (sodium / 500) else 0',
        'refined_carb_density_strong': '(carbohydrates / (fiber + 1))^2',
    },
}, 'trained_model.pkl')