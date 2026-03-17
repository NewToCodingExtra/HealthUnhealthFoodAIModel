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

#number of samples
n = 10_000

#declares core or optional features (can add more optional features later, remove this after final edit (the comment only) dev:Joshua)
core_features = ["calories", "sugar", "fat", "fiber", "protein", "sodium", "cholesterol", "saturated_fat", "carbohydrates"]
optional_features = ["vitamin_c", "calcium", "added_sugar", "trans_fat", "potassium", "magnesium", "iron", "omega3"]

healthy_calories      = np.random.uniform(20,  350,  n//2)  # kcal
healthy_sugar         = np.random.uniform(0,   20,   n//2)  # g   – natural sugar OK
healthy_fat           = np.random.uniform(0,   15,   n//2)  # g
healthy_fiber         = np.random.uniform(1.5, 15,   n//2)  # g   – tends high
healthy_protein       = np.random.uniform(2,   35,   n//2)  # g   – tends high
healthy_sodium        = np.random.uniform(0,   400,  n//2)  # mg  – tends low
healthy_cholesterol   = np.random.uniform(0,   100,  n//2)  # mg
healthy_saturated_fat = np.random.uniform(0,   5,    n//2)  # g   – tends low
healthy_carbohydrates = np.random.uniform(0,   50,   n//2)  # g   – tends moderate 

healthy_vitamin_c     = np.random.uniform(3,   90,   n//2)  # mg  – tends high
healthy_calcium       = np.random.uniform(10,  500,  n//2)  # mg
healthy_added_sugar   = np.random.uniform(0,   8,    n//2)  # g   – tends low
healthy_trans_fat     = np.random.uniform(0,   0.5,  n//2)  # g   – near zero
healthy_potassium     = np.random.uniform(150, 600,  n//2)  # mg  – tends high
healthy_magnesium     = np.random.uniform(20,  120,  n//2)  # mg  – tends high
healthy_iron          = np.random.uniform(0.5, 5,    n//2)  # mg  – tends high
healthy_omega3        = np.random.uniform(0.05,2.5,  n//2)  # g   – tends high

unhealthy_calories      = np.random.uniform(100,  700,  n//2)  # kcal – tends high
unhealthy_sugar         = np.random.uniform(8,    60,   n//2)  # g    – tends high
unhealthy_fat           = np.random.uniform(8,    45,   n//2)  # g    – tends high
unhealthy_fiber         = np.random.uniform(0,    4,    n//2)  # g    – tends low
unhealthy_protein       = np.random.uniform(0,    20,   n//2)  # g    – tends low
unhealthy_sodium        = np.random.uniform(200,  2000, n//2)  # mg   – tends high
unhealthy_cholesterol   = np.random.uniform(20,   300,  n//2)  # mg   – tends high
unhealthy_saturated_fat = np.random.uniform(4,    30,   n//2)  # g    – tends high
unhealthy_carbohydrates = np.random.uniform(5,   120,  n//2)  # g    – tends high
 
unhealthy_vitamin_c     = np.random.uniform(0,    15,   n//2)  # mg   – tends low
unhealthy_calcium       = np.random.uniform(0,    150,  n//2)  # mg
unhealthy_added_sugar   = np.random.uniform(6,    50,   n//2)  # g    – tends high
unhealthy_trans_fat     = np.random.uniform(0.5,  5,    n//2)  # g    – elevated
unhealthy_potassium     = np.random.uniform(0,    200,  n//2)  # mg   – tends low
unhealthy_magnesium     = np.random.uniform(0,    30,   n//2)  # mg   – tends low
unhealthy_iron          = np.random.uniform(0,    1.5,  n//2)  # mg   – tends low
unhealthy_omega3        = np.random.uniform(0,    0.2,  n//2)  # g    – tends low

# combining healthy and unhealthy ranges
calories      = np.concatenate([healthy_calories, unhealthy_calories]) # combining healthy and unhealthy calories
sugar         = np.concatenate([healthy_sugar, unhealthy_sugar]) # combining healthy and unhealthy sugar
fat           = np.concatenate([healthy_fat, unhealthy_fat]) # combining healthy and unhealthy fat
fiber         = np.concatenate([healthy_fiber, unhealthy_fiber]) # combining healthy and unhealthy fiber
protein       = np.concatenate([healthy_protein, unhealthy_protein]) # combining healthy and unhealthy protein
sodium        = np.concatenate([healthy_sodium, unhealthy_sodium]) # combining healthy and unhealthy sodium
cholesterol   = np.concatenate([healthy_cholesterol, unhealthy_cholesterol]) # combining healthy and unhealthy cholesterol
saturated_fat = np.concatenate([healthy_saturated_fat, unhealthy_saturated_fat]) # combining healthy and unhealthy saturated fat
carbohydrates = np.concatenate([healthy_carbohydrates, unhealthy_carbohydrates]) # combining healthy and unhealthy carbohydrates

vitamin_c   = np.concatenate([healthy_vitamin_c, unhealthy_vitamin_c]) # combining healthy and unhealthy vitamin c
calcium     = np.concatenate([healthy_calcium, unhealthy_calcium]) # combining healthy and unhealthy calcium
added_sugar = np.concatenate([healthy_added_sugar, unhealthy_added_sugar]) # combining healthy and unhealthy added sugar
trans_fat   = np.concatenate([healthy_trans_fat, unhealthy_trans_fat]) # combining healthy and unhealthy trans fat
potassium   = np.concatenate([healthy_potassium,   unhealthy_potassium])
magnesium   = np.concatenate([healthy_magnesium,   unhealthy_magnesium])
iron        = np.concatenate([healthy_iron,        unhealthy_iron])
omega3      = np.concatenate([healthy_omega3,      unhealthy_omega3])

# calculating health score based on given nutrition facts
health_score = (
      fiber * 4.0  # high fiber content is important for a healthy diet, so it gets a higher weight
    + protein * 1.5 # high protein content is important for muscle growth and repair, so it gets a higher weight
    + vitamin_c * 0.4  # high vitamin c content is important for immune function, so it gets a moderate weight
    + calcium * 0.04  # high calcium content is important for bone health, but it is not as important as other nutrition facts, so it gets a lower weight
    + potassium * 0.012  # high potassium content is important for heart health, but it is not as important as other nutrition facts, so it gets a lower weight
    + magnesium * 0.15 # high magnesium content is important for muscle and nerve
    + iron * 1.5  # high iron content is important for blood health, but it is not as important as other nutrition facts, so it gets a lower weight
    + omega3 * 6.0  # high omega-3 content is important for heart and brain health, but it is not as important as other nutrition facts, so it gets a lower weight
    - calories * 0.05  # high calorie content is unhealthy, so it gets a negative weight
    - carbohydrates * 0.15  # high carbohydrate content can be unhealthy, especially if it is refined carbohydrates, so it gets a negative weight
    - sugar * 2.5  # high sugar content is unhealthy, so it gets a negative weight
    - fat * 0.8  # high fat content is unhealthy, so it gets a negative weight
    - sodium * 0.015  # high sodium content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    - cholesterol * 0.04  # high cholesterol content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    - saturated_fat * 4.0  # high saturated fat content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    - added_sugar * 5.0  # high added sugar content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    - trans_fat * 20.0 # high trans fat content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
) # dev: Joshua : fix the comment later as it is ai generated!

#adding noise to the health score to make it more realistic and less deterministic, since in real life, the healthiness of a food item is not solely determined by its nutrition facts, but also by other factors such as portion size, cooking method, and individual dietary needs. The noise is added to simulate these real-life factors and make the model more robust and generalizable.
health_score = health_score + np.random.normal(0, 10, size=n) # adding random noise with a mean of 0 and a standard deviation of 5 to the health score
                
# generating health label based on health score
# health_label = pd.qcut(
#     health_score,
#     q=[0, 0.4, 1.0],
#     labels=['Unhealthy', 'Healthy']
# )

health_label = pd.Series(np.where(health_score > -30, 'Healthy', 'Unhealthy'))

# creating dataframe with all the nutrition facts
df = pd.DataFrame({
    "calories": calories,
    "sugar": sugar,
    "fat": fat,
    "fiber": fiber,
    "protein": protein,
    "sodium": sodium,
    "cholesterol": cholesterol,
    "saturated_fat": saturated_fat,
    "carbohydrates": carbohydrates,
    "vitamin_c": vitamin_c,
    "calcium": calcium,
    "added_sugar": added_sugar,
    "trans_fat": trans_fat,
    "potassium": potassium,
    "magnesium": magnesium,
    "iron": iron,
    "omega3": omega3,
    "health_label" : health_label
})


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