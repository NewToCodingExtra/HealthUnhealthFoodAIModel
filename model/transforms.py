import numpy as np
import pandas as pd


BASE_CORE_FEATURES = [
    "calories",
    "carbohydrates",
    "sugar",
    "fat",
    "saturated_fat",
    "sodium",
    "protein",
]

DERIVED_FEATURES = ["fried_index", "fried_starchy"]
OPTIONAL_FEATURES = ["fiber", "cholesterol", "added_sugar", "vitamin_c", "omega3"]
CORE_FEATURES = BASE_CORE_FEATURES + DERIVED_FEATURES
ALL_FEATURES = CORE_FEATURES + OPTIONAL_FEATURES
ADDED_SUGAR_IDX = ALL_FEATURES.index("added_sugar")


def add_derived_features(df_in):
    df_out = df_in.copy()
    carbs = pd.to_numeric(df_out["carbohydrates"], errors="coerce")
    fat = pd.to_numeric(df_out["fat"], errors="coerce")
    interaction = (carbs * fat) / 100.0
    df_out["fried_index"] = np.log1p(np.clip(interaction, a_min=0.0, a_max=None))
    df_out["fried_starchy"] = ((carbs > 35.0) & (fat > 8.0)).astype(float)
    return df_out


def compute_derived_features(carbohydrates, fat):
    if pd.isna(carbohydrates) or pd.isna(fat):
        return np.nan, np.nan
    interaction = (float(carbohydrates) * float(fat)) / 100.0
    fried_index = np.log1p(max(interaction, 0.0))
    fried_starchy = 1.0 if (float(carbohydrates) > 35.0 and float(fat) > 8.0) else 0.0
    return fried_index, fried_starchy


def log1p_added_sugar(X):
    x = np.asarray(X, dtype=float).copy()
    if x.ndim == 2 and x.shape[1] > ADDED_SUGAR_IDX:
        x[:, ADDED_SUGAR_IDX] = np.log1p(np.clip(x[:, ADDED_SUGAR_IDX], a_min=0.0, a_max=None))
    return x
