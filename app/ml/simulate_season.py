import os
import sys
from typing import Dict, List, Tuple

import joblib
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


def _get_actual_2025_champion() -> Tuple[str, float]:
    db: Session = SessionLocal()
    try:
        from app.models import DriverStanding
        standings = (
            db.query(DriverStanding)
            .filter(DriverStanding.season == 2025)
            .order_by(DriverStanding.round.desc())
            .all()
        )
        if not standings:
            return "Unknown", 0.0
        latest_round = max(s.round for s in standings)
        final = [s for s in standings if s.round == latest_round]
        final.sort(key=lambda x: x.points or 0, reverse=True)
        champion_id = final[0].driver_id
        champion = db.query(Driver).get(champion_id)
        champion_name = champion.full_name if champion else f"Driver {champion_id}"
        champion_points = float(final[0].points or 0)
        return champion_name, champion_points
    finally:
        db.close()


def simulate_season(n_sims: int = 10000, random_state: int = 42) -> pd.DataFrame:
    data = prepare_data()
    model = joblib.load(os.path.join(os.path.dirname(__file__), "artifacts", "winner_model.joblib"))

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

    rng = np.random.default_rng(random_state)

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

    champ_probs = sim_champions / n_sims
    avg_points = sim_points.mean(axis=0)

    rows = []
    for idx, d_id in enumerate(all_driver_ids):
        rows.append({
            "driver_id": int(d_id),
            "driver_name": driver_names.get(int(d_id), f"Driver {d_id}"),
            "wdc_probability_pct": round(float(champ_probs[idx]) * 100, 2),
            "avg_final_points": round(float(avg_points[idx]), 1),
        })

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("wdc_probability_pct", ascending=False).reset_index(drop=True)

    # Cache results for API access
    cache_path = os.path.join(os.path.dirname(__file__), "artifacts", "wdc_2025_results.json")
    results_df.to_json(cache_path, orient="records", indent=2)

    return results_df, sim_champions, sim_points, all_driver_ids


def main():
    print("=== 2025 WDC Monte Carlo Simulation (10,000 runs) ===\n")
    results_df, sim_champions, sim_points, all_driver_ids = simulate_season(n_sims=10000)

    print(results_df.to_string(index=False))

    actual_champion_name, actual_points = _get_actual_2025_champion()
    print(f"\nActual 2025 Champion: {actual_champion_name} ({actual_points:.0f} points)")

    actual_row = results_df[results_df["driver_name"] == actual_champion_name]
    if not actual_row.empty:
        actual_prob = float(actual_row.iloc[0]["wdc_probability_pct"])
        field_avg = float(results_df["wdc_probability_pct"].mean())
        print(f"Simulated WDC Probability for {actual_champion_name}: {actual_prob:.2f}%")
        print(f"Field Average WDC Probability: {field_avg:.2f}%")

        if actual_prob >= field_avg * 2:
            print(f"PASS: Actual champion has meaningfully elevated probability ({actual_prob:.2f}% vs {field_avg:.2f}% field avg).")
        else:
            print(f"NOTE: Actual champion probability ({actual_prob:.2f}%) is below 2x field average ({field_avg:.2f}%).")
    else:
        print("WARNING: Actual champion not found in simulation driver list.")


if __name__ == "__main__":
    main()