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


def diagnose():
    data = prepare_data()
    model = joblib.load(os.path.join(os.path.dirname(__file__), "artifacts", "winner_model.joblib"))

    X_val = data["X_val"]
    val_df = data["val_df"].copy()

    val_probs = model.predict_proba(X_val)[:, 1]
    val_df = val_df.reset_index(drop=True)
    val_df["win_prob"] = val_probs

    driver_names = _load_driver_names()

    # 1. Average predicted win probability per driver across all 2025 races
    avg_probs = val_df.groupby("driver_id")["win_prob"].mean().reset_index()
    avg_probs["driver_name"] = avg_probs["driver_id"].map(driver_names)
    avg_probs = avg_probs.sort_values("win_prob", ascending=False).reset_index(drop=True)

    print("=== Average Win Probability Per Driver (2025 Validation Set) ===")
    print(avg_probs.to_string(index=False))

    # 2. Per-race combined Piastri+Norris win probability share
    per_race_share = []
    for race_id, group in val_df.groupby("race_id"):
        piastri_row = group[group["driver_id"] == 7]
        norris_row = group[group["driver_id"] == 5]
        piastri_prob = float(piastri_row["win_prob"].values[0]) if not piastri_row.empty else 0.0
        norris_prob = float(norris_row["win_prob"].values[0]) if not norris_row.empty else 0.0
        combined = piastri_prob + norris_prob

        season = int(group["season"].values[0])
        round_num = int(group["round"].values[0])

        per_race_share.append({
            "season": season,
            "round": round_num,
            "piastri_prob": round(piastri_prob, 4),
            "norris_prob": round(norris_prob, 4),
            "combined_share": round(combined, 4),
        })

    share_df = pd.DataFrame(per_race_share)

    print("\n=== Per-Race Piastri + Norris Combined Win Probability Share ===")
    print(share_df.to_string(index=False))
    print(f"\nMean combined share: {share_df['combined_share'].mean():.4f}")
    print(f"Min combined share: {share_df['combined_share'].min():.4f}")
    print(f"Max combined share: {share_df['combined_share'].max():.4f}")

    # 3. Check if probabilities sum to ~1 per race (sanity check)
    race_sums = val_df.groupby("race_id")["win_prob"].sum().reset_index()
    print("\n=== Per-Race Win Probability Sums (sanity check) ===")
    print(race_sums.describe().to_string())


if __name__ == "__main__":
    diagnose()
