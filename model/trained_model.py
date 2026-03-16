import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# python -m pip install --upgrade pandas numpy matplotlib scikit-learn 
# run this on terminal

np.random.seed(42)

#number of samples
n = 10_000

#declares core or optional features (can add more optional features later, remove this after final edit (the comment only) dev:Joshua)
core_features = ["calories", "sugar", "fat", "fiber", "protein", "sodium", "cholesterol", "saturated_fat"]
optional_features = ["vitamin_c", "calcium", "added_sugar", "trans_fat"]

#Considered as healthy nutrition facts (n//2) (healthy = 5000 and unhealthy = 5000)
healthy_calories = np.random.randint(80, 400, size=n//2) #low to medium means healthy
healthy_sugar = np.random.randint(0, 25, size=n//2) #low means healthy
healthy_fat = np.random.randint(0, 18, size=n//2) #low means healthy
healthy_fiber = np.random.randint(2, 12, size=n//2) #high means healthy
healthy_protein = np.random.randint(4, 25, size=n//2) #high means healthy
healthy_sodium = np.random.randint(50, 400, size=n//2) #low means healthy
healthy_cholesterol = np.random.randint(0, 120, size=n//2) #low means healthy
healthy_saturated_fat = np.random.randint(0, 15, size=n//2) #low means healthy

healthy_vitamin_c = np.random.randint(10, 120, size=n//2) #high means healthy
healthy_calcium = np.random.randint(50, 350, size=n//2) #high means healthy
healthy_added_sugar = np.random.randint(0, 12, size=n//2) #low means healthy
healthy_trans_fat = np.random.randint(0, 2, size=n//2) #low means healthy

#Considered as unhealthy nutrition facts
unhealthy_calories = np.random.randint(200, 900, size=n//2) #high  means unhealthy
unhealthy_sugar = np.random.randint(10, 60, size=n//2) #high means unhealtthy
unhealthy_fat = np.random.randint(8, 45, size=n//2) #high means unhealthy
unhealthy_fiber = np.random.randint(0, 6, size=n//2) #low means unhealthy
unhealthy_protein = np.random.randint(0, 18, size=n//2) #low means unhealthy
unhealthy_sodium = np.random.randint(200, 2000, size=n//2) #high means unhealty
unhealthy_cholesterol = np.random.randint(40, 300, size=n//2) #high means unhealthy
unhealthy_saturated_fat = np.random.randint(6, 60, size=n//2) #high means unhealthy

unhealthy_vitamin_c = np.random.randint(0, 70, size=n//2) #low means unhealthy
unhealthy_calcium = np.random.randint(0, 200, size=n//2) #low means unhealthy
unhealthy_added_sugar = np.random.randint(5, 40, size=n//2) #high means unhealthy
unhealthy_trans_fat = np.random.randint(0, 6, size=n//2) #high means unhealthy

# combining healthy and unhealthy ranges
calories = np.concatenate([healthy_calories, unhealthy_calories]) # combining healthy and unhealthy calories
sugar = np.concatenate([healthy_sugar, unhealthy_sugar]) # combining healthy and unhealthy sugar
fat = np.concatenate([healthy_fat, unhealthy_fat]) # combining healthy and unhealthy fat
fiber = np.concatenate([healthy_fiber, unhealthy_fiber]) # combining healthy and unhealthy fiber
protein = np.concatenate([healthy_protein, unhealthy_protein]) # combining healthy and unhealthy protein
sodium = np.concatenate([healthy_sodium, unhealthy_sodium]) # combining healthy and unhealthy sodium
cholesterol = np.concatenate([healthy_cholesterol, unhealthy_cholesterol]) # combining healthy and unhealthy cholesterol
saturated_fat = np.concatenate([healthy_saturated_fat, unhealthy_saturated_fat]) # combining healthy and unhealthy saturated fat

vitamin_c = np.concatenate([healthy_vitamin_c, unhealthy_vitamin_c]) # combining healthy and unhealthy vitamin c
calcium = np.concatenate([healthy_calcium, unhealthy_calcium]) # combining healthy and unhealthy calcium
added_sugar = np.concatenate([healthy_added_sugar, unhealthy_added_sugar]) # combining healthy and unhealthy added sugar
trans_fat = np.concatenate([healthy_trans_fat, unhealthy_trans_fat]) # combining healthy and unhealthy trans fat

# calculating health score based on given nutrition facts
health_score = (
    fiber * 0.25 + # high fiber content is important for a healthy diet, so it gets a higher weight
    protein * 0.2 + # high protein content is important for muscle growth and repair, so it gets a higher weight
    vitamin_c * 0.1 + # high vitamin c content is important for immune function, so it gets a moderate weight
    calcium * 0.05 + # high calcium content is important for bone health, but it is not as important as other nutrition facts, so it gets a lower weight
    -calories * 0.1 - # high calorie content is unhealthy, so it gets a negative weight
    -sugar * 0.2 - # high sugar content is unhealthy, so it gets a negative weight
    -fat * 0.15 - # high fat content is unhealthy, so it gets a negative weight
    -sodium * 0.05 - # high sodium content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    -cholesterol * 0.05 - # high cholesterol content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    -saturated_fat * 0.2 - # high saturated fat content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    -added_sugar * 0.05 - # high added sugar content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
    -trans_fat * 0.25 # high trans fat content is unhealthy, but it is not as important as other nutrition facts, so it gets a lower weight
) # dev: Joshua : fix the comment later as it is ai generated!

#adding noise to the health score to make it more realistic and less deterministic, since in real life, the healthiness of a food item is not solely determined by its nutrition facts, but also by other factors such as portion size, cooking method, and individual dietary needs. The noise is added to simulate these real-life factors and make the model more robust and generalizable.
health_score = health_score + np.random.normal(0, 5, size=n) # adding random noise with a mean of 0 and a standard deviation of 5 to the health score
                
# generating health label based on health score
health_label = pd.qcut(
    health_score,
    q=2,
    labels=['Unhealthy', 'Healthy']
)

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
    "vitamin_c": vitamin_c,
    "calcium": calcium,
    "added_sugar": added_sugar,
    "trans_fat": trans_fat,
    "health_label" : health_label
})


encoder = LabelEncoder()
#convert label into numeric values (unhealthy = 0 and healthy = 1)
df['health_label'] = encoder.fit_transform(df['health_label'])

#separating core features and optional features
#only core features
X_core = df[core_features] #no need to use df.drop() since we are only selecting core features only excluding health_label                        

#all features (core + optional)
X_all = df[core_features + optional_features] #no need to use df.drop() since we are selecting all features including core and optional features excluding health_label                    

#target variable (unhealthy = 0 and healthy = 1)
y = df['health_label']

scaler_core = StandardScaler()
scaler_all = StandardScaler()

X_core_scaled = scaler_core.fit_transform(X_core) # scale the core features
X_all_scaled = scaler_all.fit_transform(X_all) # scale all features (core + optional)

#training model using only the core features first

#splitting the data into training and testing sets (80% training and 20% testing) (only core features   and target variable)
X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(
    X_core_scaled, y, test_size=0.2, random_state=42
)

# this line creates a logistic regression model with a maximum of 1000 iterations, this is how we train our model to make predictions based on the data we give it.
# this model is strong if optional features are not filled, but it is weak if optional features are filled, since it relies on core features to make predictions, so if optional features are filled, it will not perform well.                     
model_core = LogisticRegression(max_iter=1000)

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
model_all = LogisticRegression(max_iter=1000)

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

joblib.dump({'models': trained_models, 'scalers': trained_scalers}, 'trained_model.pkl')