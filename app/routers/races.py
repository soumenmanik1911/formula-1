from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.ingestion import jolpica_client

router = APIRouter(prefix="/races", tags=["races"])


@router.get("/{season}")
def get_races(season: int, db: Session = Depends(get_db)):
    races = jolpica_client.get_season_races(season)
    if not races:
        raise HTTPException(status_code=404, detail=f"No races found for season {season}")
    result = []
    for race in races:
        results = []
        for rr in race.results:
            results.append({
                "position": rr.position,
                "driver": rr.driver.full_name if rr.driver else None,
                "constructor": rr.constructor.name if rr.constructor else None,
                "points": rr.points,
                "status": rr.status,
                "grid": rr.grid,
                "laps": rr.laps,
                "time": rr.time,
                "time_ms": rr.time_ms,
            })
        result.append({
            "season": race.season,
            "round": race.round,
            "race_name": race.race_name,
            "date": race.date,
            "circuit_name": race.circuit_name,
            "country": race.country,
            "results": results,
        })
    return result
