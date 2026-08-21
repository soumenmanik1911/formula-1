import os
import sys
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.database import SessionLocal
from app.models import Driver
from app.ml.prepare_data import prepare_data


def _load_driver_names() -> Dict[int, str]:
    db: Session = SessionLocal()
    try:
        drivers = db.query(Driver).all()
        return {d.id: d.full_name for d in drivers}
    finally:
        db.close()


def check():
    data = prepare_data()
    model = joblib.load(os.path.join(os.path.dirname(__file__), "artifacts", "podium_model.joblib"))

    X_val = data["X_val"]
    val_df = data["val_df"].copy()

    val_probs = model.predict_proba(X_val)[:, 1]
    val_df = val_df.reset_index(drop=True)
    val_df["podium_prob"] = val_probs

    driver_names = _load_driver_names()

    avg_probs = val_df.groupby("driver_id")["podium_prob"].mean().reset_index()
    avg_probs["driver_name"] = avg_probs["driver_id"].map(driver_names)
    avg_probs = avg_probs.sort_values("podium_prob", ascending=False).reset_index(drop=True)

    print("=== Average Podium Probability Per Driver (2025) ===")
    print(avg_probs.to_string(index=False))

    print("\n=== Per-Race Podium Probability Sums ===")
    race_sums = val_df.groupby("race_id")["podium_prob"].sum()
    print(race_sums.describe().to_string())
    print(f"Mean sum: {race_sums.mean():.4f}")
    print(f"Min sum: {race_sums.min():.4f}")
    print(f"Max sum: {race_sums.max():.4f}")


if __name__ == "__main__":
    check()

