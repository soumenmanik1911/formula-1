from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.ingestion import jolpica_client

router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("/drivers")
def get_driver_standings(season: int = Query(..., ge=2018, le=2030), db: Session = Depends(get_db)):
    standings = jolpica_client.get_driver_standings(season)
    if not standings:
        raise HTTPException(status_code=404, detail=f"No driver standings found for season {season}")
    return [
        {
            "position": s.position,
            "driver_id": s.driver.driver_id,
            "full_name": s.driver.full_name,
            "nationality": s.driver.nationality,
            "points": s.points,
            "wins": s.wins,
        }
        for s in standings
    ]


@router.get("/constructors")
def get_constructor_standings(season: int = Query(..., ge=2018, le=2030), db: Session = Depends(get_db)):
    standings = jolpica_client.get_constructor_standings(season)
    if not standings:
        raise HTTPException(status_code=404, detail=f"No constructor standings found for season {season}")
    return [
        {
            "position": s.position,
            "constructor_id": s.constructor.constructor_id,
            "name": s.constructor.name,
            "nationality": s.constructor.nationality,
            "points": s.points,
            "wins": s.wins,
        }
        for s in standings
    ]
