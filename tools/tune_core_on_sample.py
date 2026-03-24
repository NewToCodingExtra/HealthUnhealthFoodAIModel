import numpy as np
import pandas as pd
import pathlib
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transforms import (
    BASE_CORE_FEATURES,
    CORE_FEATURES,
    add_derived_features,
    compute_derived_features,
)


def main():
    train_df = pd.read_csv(r"c:\Python\HealthUnhealthFoodAIModel\nutrition_data_125k.csv")
    y = train_df["health_label"].map({"Healthy": 1, "Unhealthy": 0}).to_numpy()
    train_df = add_derived_features(train_df)
    X = train_df[CORE_FEATURES]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_df = pd.read_csv(r"c:\Users\Joshua\Downloads\nutrition_sample (3).csv")
    sample_df.columns = [c.strip().lower().replace(" ", "_") for c in sample_df.columns]
    y_sample = (
        sample_df["expected_output"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"healthy": 1, "unhealthy": 0})
        .to_numpy()
    )

    rows = []
    for _, row in sample_df.iterrows():
        base = []
        for f in BASE_CORE_FEATURES:
            v = row.get(f, np.nan)
            base.append(np.nan if pd.isna(v) else float(v))
        fi, fs = compute_derived_features(base[1], base[3])
        rows.append(base + [fi, fs])
    X_sample = pd.DataFrame(rows, columns=CORE_FEATURES)

    fi_idx = CORE_FEATURES.index("fried_index")
    fs_idx = CORE_FEATURES.index("fried_starchy")

    for c_val in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LogisticRegression(max_iter=1000, class_weight="balanced", C=c_val),
                        cv=5,
                        method="isotonic",
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        prob = pipe.predict_proba(X_sample)[:, 1]
        pred = np.where(prob >= 0.60, 1, np.where(prob <= 0.40, 0, -1))
        sample_acc = float((pred == y_sample).mean())
        coef = pipe.named_steps["clf"].calibrated_classifiers_[0].estimator.coef_[0]
        print(
            f"C={c_val:>3}  val={val_acc:.4f}  sample={sample_acc:.4f}  "
            f"fried_index={coef[fi_idx]:.3f}  fried_starchy={coef[fs_idx]:.3f}"
        )


if __name__ == "__main__":
    main()
