import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

core = ["calories", "carbohydrates", "sugar", "fat", "saturated_fat", "sodium", "protein"]
derived = ["fried_index", "fried_starchy"]
optional = ["fiber", "cholesterol", "added_sugar", "vitamin_c", "omega3"]
all_features = core + derived + optional


def add_derived(df):
    out = df.copy()
    carbs = pd.to_numeric(out["carbohydrates"], errors="coerce")
    fat = pd.to_numeric(out["fat"], errors="coerce")
    out["fried_index"] = np.log1p(np.clip((carbs * fat) / 100.0, a_min=0.0, a_max=None))
    out["fried_starchy"] = ((carbs > 35.0) & (fat > 8.0)).astype(float)
    return out


def log1p_added_sugar(X):
    arr = np.asarray(X, dtype=float).copy()
    idx = all_features.index("added_sugar")
    arr[:, idx] = np.log1p(np.clip(arr[:, idx], a_min=0.0, a_max=None))
    return arr


def main():
    df = pd.read_csv(r"c:\Python\HealthUnhealthFoodAIModel\nutrition_data_125k.csv")
    y = df["health_label"].map({"Healthy": 1, "Unhealthy": 0}).to_numpy()
    df = add_derived(df)
    X = df[all_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for cw in [None, "balanced"]:
        for c_val in [0.3, 0.5, 0.7, 1.0]:
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("log1p_added_sugar", FunctionTransformer(log1p_added_sugar, validate=False)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        CalibratedClassifierCV(
                            LogisticRegression(max_iter=1000, class_weight=cw, C=c_val),
                            cv=5,
                            method="isotonic",
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, pred)
            coef = pipe.named_steps["clf"].calibrated_classifiers_[0].estimator.coef_[0]
            print(
                f"cw={cw}, C={c_val}, acc={acc:.4f}, "
                f"added_sugar={coef[all_features.index('added_sugar')]:.4f}, "
                f"fried_index={coef[all_features.index('fried_index')]:.4f}, "
                f"fried_starchy={coef[all_features.index('fried_starchy')]:.4f}"
            )


if __name__ == "__main__":
    main()
