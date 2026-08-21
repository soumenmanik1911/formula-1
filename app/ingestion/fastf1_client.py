import os
from typing import Optional
import fastf1
from fastf1 import get_session
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Race, LapData

fastf1.Cache.enable_cache(os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))


def _get_db() -> Session:
    return SessionLocal()


def _parse_lap_time_ms(lap_time) -> Optional[int]:
    """Convert a pandas Timedelta or string lap time to total milliseconds."""
    if lap_time is None:
        return None
    try:
        td = pd.Timedelta(lap_time)
        return int(td.total_seconds() * 1000)
    except Exception:
        try:
            # Try string format like "1:23.456"
            parts = str(lap_time).split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return int((minutes * 60 + seconds) * 1000)
        except Exception:
            pass
    return None


def _parse_session_time_ms(time_str) -> Optional[int]:
    """Convert a session time string like '1:23:06.801' to total milliseconds."""
    if not time_str:
        return None
    try:
        parts = str(time_str).split(":")
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return int((hours * 3600 + minutes * 60 + seconds) * 1000)
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return int((minutes * 60 + seconds) * 1000)
    except Exception:
        pass
    return None


import pandas as pd


def ingest_fastf1_session(season: int, round_num: int) -> dict:
    """
    Pull lap times and basic session data for a given season/round using FastF1.
    Persists lap data into the LapData table. Returns a summary dict.
    """
    db = _get_db()
    try:
        session = get_session(season, round_num, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        event = session.event
        race = db.query(Race).filter_by(season=season, round=round_num).first()
        if not race:
            race = Race(
                season=season,
                round=round_num,
                race_name=event["EventName"],
                date=str(event["EventDate"])[:10] if event.get("EventDate") else None,
                circuit_name=event.get("CircuitName"),
                country=event.get("Country"),
            )
            db.add(race)
            db.flush()

        laps = session.laps
        if laps is None or laps.empty:
            return {
                "season": season,
                "round": round_num,
                "event_name": event["EventName"],
                "session": "Race",
                "laps_loaded": 0,
                "drivers": [],
            }

        drivers = sorted(laps["Driver"].unique().tolist())
        total_laps = int(laps["LapNumber"].max()) if "LapNumber" in laps.columns else 0

        # Clear existing lap data for this race to avoid duplicates on re-run
        db.query(LapData).filter_by(race_id=race.id).delete()

        for _, lap in laps.iterrows():
            lap_data = LapData(
                race_id=race.id,
                driver_code=str(lap.get("Driver", "")),
                lap_number=int(lap.get("LapNumber", 0)) if lap.get("LapNumber") else None,
                lap_time=str(lap.get("LapTime")) if lap.get("LapTime") is not None else None,
                lap_time_ms=_parse_lap_time_ms(lap.get("LapTime")),
                tyre_compound=str(lap.get("Compound")) if lap.get("Compound") is not None else None,
                tyre_age=int(lap.get("TyreAge")) if lap.get("TyreAge") is not None else None,
                stint=int(lap.get("Stint")) if lap.get("Stint") is not None else None,
                fresh_tyre=str(lap.get("FreshTyre")) if lap.get("FreshTyre") is not None else None,
                track_status=str(lap.get("TrackStatus")) if lap.get("TrackStatus") is not None else None,
                lap_start_time=str(lap.get("LapStartTime")) if lap.get("LapStartTime") is not None else None,
                driver_number=int(lap.get("DriverNumber")) if lap.get("DriverNumber") is not None else None,
                team=str(lap.get("Team")) if lap.get("Team") is not None else None,
            )
            db.add(lap_data)

        db.commit()
        return {
            "season": season,
            "round": round_num,
            "event_name": event["EventName"],
            "session": "Race",
            "laps_loaded": total_laps,
            "drivers": drivers,
        }
    except Exception as e:
        db.rollback()
        raise RuntimeError(f"FastF1 ingestion failed for {season} round {round_num}: {e}")
    finally:
        db.close()


def get_cached_session_summary(season: int, round_num: int) -> Optional[dict]:
    """Check if FastF1 cache has data for this session by attempting a lightweight load."""
    try:
        session = get_session(season, round_num, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        return {
            "season": season,
            "round": round_num,
            "event_name": session.event["EventName"],
            "cached": True,
        }
    except Exception:
        return None
