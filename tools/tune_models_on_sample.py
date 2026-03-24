import pathlib
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transforms import (
    BASE_CORE_FEATURES,
    CORE_FEATURES,
    OPTIONAL_FEATURES,
    ALL_FEATURES,
    add_derived_features,
    compute_derived_features,
    log1p_added_sugar,
)


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


def verdict_from_prob(p, hi=0.60, lo=0.40):
    if p >= hi:
        return 1
    if p <= lo:
        return 0
    return -1


def build_sample_xy(path_csv):
    df = pd.read_csv(path_csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    y = df["expected_output"].astype(str).str.strip().str.lower().map({"healthy": 1, "unhealthy": 0}).to_numpy()

    core_rows = []
    all_rows = []
    for _, row in df.iterrows():
        base = []
        for f in BASE_CORE_FEATURES:
            v = row.get(f, np.nan)
            base.append(np.nan if pd.isna(v) else float(v))
        fi, fs = compute_derived_features(base[1], base[3])
        core_vals = base + [fi, fs]
        all_vals = core_vals.copy()
        for f in OPTIONAL_FEATURES:
            v = row.get(f, np.nan)
            all_vals.append(np.nan if pd.isna(v) else float(v))
        core_rows.append(core_vals)
        all_rows.append(all_vals)
    return pd.DataFrame(core_rows, columns=CORE_FEATURES), pd.DataFrame(all_rows, columns=ALL_FEATURES), y


def main():
    train = pd.read_csv(r"c:\Python\HealthUnhealthFoodAIModel\nutrition_data_125k.csv")
    train = add_derived_features(train)
    y = train["health_label"].map({"Healthy": 1, "Unhealthy": 0}).to_numpy()
    X_core = train[CORE_FEATURES]
    X_all = train[ALL_FEATURES]

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_core, y, test_size=0.2, random_state=42, stratify=y)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

    sample_core, sample_all, y_sample = build_sample_xy(r"c:\Users\Joshua\Downloads\nutrition_sample (3).csv")

    best = None
    for method in ["sigmoid", "isotonic"]:
        for cw in [None, "balanced"]:
            for c in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                core = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", CalibratedClassifierCV(
                        LogisticRegression(max_iter=1000, class_weight=cw, C=c),
                        cv=5,
                        method=method,
                    )),
                ])
                allm = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("log1p_added_sugar", FunctionTransformer(log1p_added_sugar, validate=False)),
                    ("scaler", StandardScaler()),
                    ("clf", CalibratedClassifierCV(
                        LogisticRegression(max_iter=1000, class_weight=cw, C=c),
                        cv=5,
                        method=method,
                    )),
                ])
                core.fit(Xc_tr, yc_tr)
                allm.fit(Xa_tr, ya_tr)

                pc = core.predict_proba(sample_core)[:, 1]
                pa = allm.predict_proba(sample_all)[:, 1]
                vc = np.array([verdict_from_prob(p) for p in pc])
                va = np.array([verdict_from_prob(p) for p in pa])
                core_acc = (vc == y_sample).mean()
                all_acc = (va == y_sample).mean()
                all_brier = brier_score_loss(y_sample, pa)
                all_ece = ece_percent(y_sample, pa)

                # prioritize sample accuracy and calibration of all-features model
                score = (all_acc * 10) - (all_brier * 2) - (all_ece / 100.0)
                row = (score, method, cw, c, core_acc, all_acc, all_brier, all_ece)
                if best is None or row[0] > best[0]:
                    best = row
                print(
                    f"method={method:8} cw={str(cw):8} C={c:<3} "
                    f"core={core_acc*100:5.2f}% all={all_acc*100:5.2f}% "
                    f"all_brier={all_brier:.4f} all_ece={all_ece:.2f}%"
                )

    print("\nBEST:")
    print(best)


if __name__ == "__main__":
    main()
