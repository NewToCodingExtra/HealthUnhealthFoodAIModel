import pathlib
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transforms import BASE_CORE_FEATURES, CORE_FEATURES, add_derived_features, compute_derived_features


def ece_percent(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if not np.any(m):
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += abs(acc - conf) * (m.sum() / n)
    return ece * 100


def verdicts(prob, hi=0.60, lo=0.40):
    p = np.asarray(prob)
    return np.where(p >= hi, 1, np.where(p <= lo, 0, -1))


def sample_xy(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    y = df["expected_output"].astype(str).str.strip().str.lower().map({"healthy": 1, "unhealthy": 0}).to_numpy()
    rows = []
    for _, row in df.iterrows():
        base = []
        for f in BASE_CORE_FEATURES:
            v = row.get(f, np.nan)
            base.append(np.nan if pd.isna(v) else float(v))
        fi, fs = compute_derived_features(base[1], base[3])
        rows.append(base + [fi, fs])
    X = pd.DataFrame(rows, columns=CORE_FEATURES)
    return X, y


def build_pipe(cfg):
    base_lr = LogisticRegression(
        max_iter=1500,
        class_weight=cfg["class_weight"],
        C=cfg["C"],
        penalty=cfg["penalty"],
        solver=cfg["solver"],
        l1_ratio=cfg.get("l1_ratio"),
    )

    if cfg["calibration"] == "none":
        clf = base_lr
    else:
        clf = CalibratedClassifierCV(base_lr, cv=cfg["cv"], method=cfg["calibration"])

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def main():
    df = pd.read_csv(r"c:\Python\HealthUnhealthFoodAIModel\nutrition_data_125k.csv")
    df = add_derived_features(df)
    y = df["health_label"].map({"Healthy": 1, "Unhealthy": 0}).to_numpy()
    X = df[CORE_FEATURES]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_sample, y_sample = sample_xy(r"c:\Users\Joshua\Downloads\nutrition_sample (3).csv")

    configs = []
    for class_weight in [None, "balanced"]:
        for C in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            configs.append(
                {
                    "class_weight": class_weight,
                    "C": C,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "calibration": "isotonic",
                    "cv": 5,
                }
            )
            configs.append(
                {
                    "class_weight": class_weight,
                    "C": C,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "calibration": "sigmoid",
                    "cv": 5,
                }
            )
            configs.append(
                {
                    "class_weight": class_weight,
                    "C": C,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "calibration": "none",
                    "cv": 0,
                }
            )

    best = None
    for cfg in configs:
        pipe = build_pipe(cfg)
        pipe.fit(X_train, y_train)

        # validation
        if cfg["calibration"] == "none":
            val_prob = pipe.predict_proba(X_val)[:, 1]
        else:
            val_prob = pipe.predict_proba(X_val)[:, 1]
        val_pred = verdicts(val_prob)
        val_acc = float((val_pred == y_val).mean())

        # sample
        sample_prob = pipe.predict_proba(X_sample)[:, 1]
        sample_pred = verdicts(sample_prob)
        sample_acc = float((sample_pred == y_sample).mean())
        brier = float(brier_score_loss(y_sample, sample_prob))
        ece = float(ece_percent(y_sample, sample_prob))
        borderlines = int((sample_pred == -1).sum())

        # prioritize sample accuracy, then calibration, then validation
        score = (sample_acc * 100.0) - (brier * 8.0) - (ece * 0.05) + (val_acc * 10.0)
        row = (score, cfg, val_acc, sample_acc, brier, ece, borderlines)
        if best is None or row[0] > best[0]:
            best = row

        print(
            f"cal={cfg['calibration']:8} cw={str(cfg['class_weight']):8} C={cfg['C']:<4} "
            f"val={val_acc*100:5.2f}% sample={sample_acc*100:5.2f}% "
            f"brier={brier:.4f} ece={ece:5.2f}% borderline={borderlines}"
        )

    print("\nBEST CONFIG")
    print(best)


if __name__ == "__main__":
    main()
