import os
import sys
from typing import Dict, Tuple

# Allow running this script directly from the project root or app/ml/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine

from app.database import DATABASE_URL

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "f1_dashboard.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

FEATURE_COLS = [
    "qualifying_position",
    "rolling_avg_finish_last5",
    "rolling_avg_points_last5",
    "constructor_rolling_avg_points_last5",
    "driver_elo",
    "circuit_avg_finish",
    "points_before_race",
    "standing_position_before_race",
]

TARGET_COL = "finish_position"

TRAIN_SEASONS = {2022, 2023, 2024}
VAL_SEASON = 2025
HOLDOUT_SEASON = 2026


def prepare_data() -> Dict:
    df = pd.read_sql("SELECT * FROM driver_race_features", ENGINE)

    df = df.dropna(subset=[TARGET_COL])

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df = df[df["season"] == VAL_SEASON].copy()
    holdout_df = df[df["season"] == HOLDOUT_SEASON].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values

    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL].values

    X_holdout = holdout_df[FEATURE_COLS]
    y_holdout = holdout_df[TARGET_COL].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_holdout_imp = imputer.transform(X_holdout)

    X_train_imp = pd.DataFrame(X_train_imp, columns=FEATURE_COLS, index=X_train.index)
    X_val_imp = pd.DataFrame(X_val_imp, columns=FEATURE_COLS, index=X_val.index)
    X_holdout_imp = pd.DataFrame(X_holdout_imp, columns=FEATURE_COLS, index=X_holdout.index)

    print(f"Train shape: {X_train_imp.shape} | Val shape: {X_val_imp.shape} | Holdout shape: {X_holdout_imp.shape}")
    print(f"Feature columns: {FEATURE_COLS}")

    return {
        "X_train": X_train_imp,
        "y_train": y_train,
        "X_val": X_val_imp,
        "y_val": y_val,
        "X_holdout": X_holdout_imp,
        "y_holdout": y_holdout,
        "imputer": imputer,
        "feature_cols": FEATURE_COLS,
        "train_df": train_df,
        "val_df": val_df,
        "holdout_df": holdout_df,
    }


if __name__ == "__main__":
    data = prepare_data()
    print("Preparation complete.")
