from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import DriverRaceFeature, Driver, Race

router = APIRouter(prefix="/features", tags=["features"])



@router.get("/{season}/{round}")
def get_features(season: int, round: int, db: Session = Depends(get_db)):
    features = (
        db.query(DriverRaceFeature)
        .filter_by(season=season, round=round)
        .order_by(DriverRaceFeature.finish_position)
        .all()
    )
    if not features:
        raise HTTPException(status_code=404, detail=f"No features found for season {season} round {round}")
    result = []
    for f in features:
        driver = db.query(Driver).get(f.driver_id)
        race = db.query(Race).get(f.race_id)
        result.append({
            "driver_id": f.driver_id,
            "driver_name": driver.full_name if driver else None,
            "race_id": f.race_id,
            "race_name": race.race_name if race else None,
            "qualifying_position": f.qualifying_position,
            "rolling_avg_finish_last5": f.rolling_avg_finish_last5,
            "rolling_avg_points_last5": f.rolling_avg_points_last5,
            "constructor_rolling_avg_points_last5": f.constructor_rolling_avg_points_last5,
            "driver_elo": f.driver_elo,
            "circuit_avg_finish": f.circuit_avg_finish,
            "points_before_race": f.points_before_race,
            "standing_position_before_race": f.standing_position_before_race,
            "finish_position": f.finish_position,
        })
    return result
