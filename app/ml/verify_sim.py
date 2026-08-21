import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.database import SessionLocal
from app.models import Driver
from app.ml.prepare_data import prepare_data
from app.ml.utils import apply_race_softmax

F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def _load_driver_names() -> Dict[int, str]:
    db: Session = SessionLocal()
    try:
        drivers = db.query(Driver).all()
        return {d.id: d.full_name for d in drivers}
    finally:
        db.close()


def verify(n_sims: int = 100):
    data = prepare_data()
    model = __import__("joblib").load(
        os.path.join(os.path.dirname(__file__), "artifacts", "winner_model.joblib")
    )

    X_val = data["X_val"]
    val_df = data["val_df"].copy()

    raw_scores = model.predict(X_val, output_margin=True)
    val_df = apply_race_softmax(val_df, raw_scores, prob_col="win_prob")

    driver_names = _load_driver_names()

    races: List[Tuple[np.ndarray, np.ndarray]] = []
    for race_id, group in val_df.groupby("race_id"):
        d_ids = group["driver_id"].values.astype(int)
        w_probs = group["win_prob"].values.astype(float)
        races.append((d_ids, w_probs))

    all_driver_ids = np.array(sorted(val_df["driver_id"].unique()))
    driver_id_to_idx = {d_id: idx for idx, d_id in enumerate(all_driver_ids)}
    n_drivers = len(all_driver_ids)

    rng = np.random.default_rng(42)

    sim_champions = np.zeros(n_drivers, dtype=int)
    sim_points = np.zeros((n_sims, n_drivers), dtype=int)

    for sim in range(n_sims):
        season_points = np.zeros(n_drivers, dtype=int)

        for d_ids, w_probs in races:
            full_probs = np.zeros(n_drivers, dtype=float)
            for i, d_id in enumerate(d_ids):
                full_probs[driver_id_to_idx[d_id]] = w_probs[i]

            remaining = np.ones(n_drivers, dtype=bool)
            for pos_idx in range(len(F1_POINTS)):
                if not remaining.any():
                    break
                rem_probs = full_probs * remaining
                rem_total = rem_probs.sum()
                if rem_total == 0:
                    break
                rem_probs = rem_probs / rem_total
                chosen_idx = rng.choice(n_drivers, p=rem_probs)
                season_points[chosen_idx] += F1_POINTS[pos_idx]
                remaining[chosen_idx] = False

        sim_points[sim] = season_points
        champion_idx = int(np.argmax(season_points))
        sim_champions[champion_idx] += 1

    # Print championship counts
    print(f"=== {n_sims} Simulation Championship Counts ===")
    for idx, d_id in enumerate(all_driver_ids):
        if sim_champions[idx] > 0:
            print(f"{driver_names.get(int(d_id), 'Driver '+str(d_id))}: {sim_champions[idx]} ({sim_champions[idx]/n_sims*100:.2f}%)")

    # Print a few specific simulation results
    print("\n=== Sample Simulation Point Totals (first 5 sims) ===")
    for sim in range(min(5, n_sims)):
        print(f"Sim {sim+1}: ", end="")
        sorted_idx = np.argsort(-sim_points[sim])
        for i in sorted_idx[:5]:
            print(f"{driver_names.get(int(all_driver_ids[i]), '?')}={sim_points[sim][i]}", end="  ")
        print()


if __name__ == "__main__":
    verify(100)
