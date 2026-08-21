from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
import os
import sys
import numpy as np
import pandas as pd
import joblib

from app.database import SessionLocal, get_db
from app.models import Driver, DriverRaceFeature
from app.ml.prepare_data import prepare_data, FEATURE_COLS
from app.ml.utils import apply_race_softmax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

router = APIRouter(prefix="/predict", tags=["predict"])


def _load_driver_names() -> dict:
    db = SessionLocal()
    try:
        return {d.id: d.full_name for d in db.query(Driver).all()}
    finally:
        db.close()


@router.get("/race/{season}/{race_round}")
def predict_race(season: int, race_round: int):
    db: Session = SessionLocal()
    try:
        race = (
            db.query(DriverRaceFeature)
            .filter(DriverRaceFeature.season == season)
            .filter(DriverRaceFeature.round == race_round)
            .all()
        )
        if not race:
            raise HTTPException(status_code=404, detail=f"No feature data for season {season} round {race_round}")

        data = prepare_data()
        imputer = data["imputer"]

        winner_model = joblib.load(
            os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "winner_model.joblib")
        )
        podium_model = joblib.load(
            os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "podium_model.joblib")
        )

        # Build DataFrame for this race
        rows = []
        for r in race:
            rows.append({
                "race_id": r.race_id,
                "driver_id": r.driver_id,
                "qualifying_position": r.qualifying_position,
                "rolling_avg_finish_last5": r.rolling_avg_finish_last5,
                "rolling_avg_points_last5": r.rolling_avg_points_last5,
                "constructor_rolling_avg_points_last5": r.constructor_rolling_avg_points_last5,
                "driver_elo": r.driver_elo,
                "circuit_avg_finish": r.circuit_avg_finish,
                "points_before_race": r.points_before_race,
                "standing_position_before_race": r.standing_position_before_race,
            })
        race_df = pd.DataFrame(rows)

        # Impute using the same imputer fitted on train data
        X_race = imputer.transform(race_df[FEATURE_COLS])
        X_race_df = pd.DataFrame(X_race, columns=FEATURE_COLS)

        # Winner probabilities: raw margins + softmax
        raw_scores = winner_model.predict(X_race_df, output_margin=True)
        race_df = apply_race_softmax(race_df, raw_scores, prob_col="win_prob")

        # Podium probabilities: binary predict_proba, normalized to sum 3.0 per race
        podium_raw = podium_model.predict_proba(X_race_df)[:, 1]
        race_df["podium_prob_raw"] = podium_raw
        for race_id, group in race_df.groupby("race_id"):
            s = group["podium_prob_raw"].sum()
            if s > 0:
                race_df.loc[group.index, "podium_prob"] = group["podium_prob_raw"] * 3.0 / s
            else:
                race_df.loc[group.index, "podium_prob"] = 0.0

        driver_names = _load_driver_names()

        result = []
        for _, row in race_df.iterrows():
            result.append({
                "driver_id": int(row["driver_id"]),
                "driver_name": driver_names.get(int(row["driver_id"]), f"Driver {row['driver_id']}"),
                "win_probability": round(float(row["win_prob"]), 6),
                "podium_probability": round(float(row["podium_prob"]), 6),
            })

        result.sort(key=lambda x: x["win_probability"], reverse=True)

        win_sum = sum(r["win_probability"] for r in result)
        podium_sum = sum(r["podium_probability"] for r in result)

        return {
            "season": season,
            "round": race_round,
            "predictions": result,
            "win_prob_sum": round(win_sum, 6),
            "podium_prob_sum": round(podium_sum, 6),
        }
    finally:
        db.close()


@router.get("/wdc/{season}")
def predict_wdc(season: int):
    if season != 2025:
        raise HTTPException(status_code=400, detail="WDC simulation currently only supported for 2025")

    from app.ml.simulate_season import simulate_season
    results_df, _, _, _ = simulate_season(n_sims=10000)

    return {
        "season": season,
        "wdc_probabilities": results_df.to_dict(orient="records"),
    }




