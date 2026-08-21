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
from app.ml.utils import apply_race_softmax


def _load_driver_names() -> Dict[int, str]:
    db: Session = SessionLocal()
    try:
        drivers = db.query(Driver).all()
        return {d.id: d.full_name for d in drivers}
    finally:
        db.close()


def check():
    data = prepare_data()
    model = joblib.load(os.path.join(os.path.dirname(__file__), "artifacts", "winner_model.joblib"))

    X_val = data["X_val"]
    val_df = data["val_df"].copy()

    raw_scores = model.predict(X_val, output_margin=True)
    val_df = apply_race_softmax(val_df, raw_scores, prob_col="win_prob")

    driver_names = _load_driver_names()

    avg_probs = val_df.groupby("driver_id")["win_prob"].mean().reset_index()
    avg_probs["driver_name"] = avg_probs["driver_id"].map(driver_names)
    avg_probs = avg_probs.sort_values("win_prob", ascending=False).reset_index(drop=True)

    print("=== Average Softmaxed Win Probability Per Driver (2025) ===")
    print(avg_probs.to_string(index=False))

    per_race_share = []
    for race_id, group in val_df.groupby("race_id"):
        piastri_row = group[group["driver_id"] == 7]
        norris_row = group[group["driver_id"] == 5]
        verstappen_row = group[group["driver_id"] == 6]
        piastri_prob = float(piastri_row["win_prob"].values[0]) if not piastri_row.empty else 0.0
        norris_prob = float(norris_row["win_prob"].values[0]) if not norris_row.empty else 0.0
        verstappen_prob = float(verstappen_row["win_prob"].values[0]) if not verstappen_row.empty else 0.0
        combined = piastri_prob + norris_prob

        season = int(group["season"].values[0])
        round_num = int(group["round"].values[0])

        per_race_share.append({
            "season": season,
            "round": round_num,
            "piastri_prob": round(piastri_prob, 4),
            "norris_prob": round(norris_prob, 4),
            "verstappen_prob": round(verstappen_prob, 4),
            "combined_share": round(combined, 4),
        })

    share_df = pd.DataFrame(per_race_share)
    print("\n=== Per-Race Top-3 Probabilities ===")
    print(share_df.to_string(index=False))
    print(f"\nMean Piastri+Norris share: {share_df['combined_share'].mean():.4f}")
    print(f"Mean Verstappen share: {share_df['verstappen_prob'].mean():.4f}")


if __name__ == "__main__":
    check()
