import requests
from typing import Optional
from sqlalchemy.orm import Session

from app.database import SessionLocal, engine, Base
from sqlalchemy.orm import joinedload
from app.models import (
    Driver, Constructor, Race, RaceResult, DriverStanding, ConstructorStanding
)

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

Base.metadata.create_all(bind=engine)


def _get_db() -> Session:
    return SessionLocal()


def _fetch(url: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def ingest_season_schedule(season: int) -> int:
    """Fetch a season's full race schedule and persist races. Returns number of races stored."""
    db = _get_db()
    try:
        data = _fetch(f"{JOLPICA_BASE}/{season}.json")
        races_data = (
            data.get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [])
        )

        stored = 0
        for r in races_data:
            circuit = r.get("Circuit", {})
            race = Race(
                season=season,
                round=int(r.get("round", 0)),
                race_name=r.get("raceName"),
                date=r.get("date"),
                circuit_name=circuit.get("circuitName"),
                country=circuit.get("Location", {}).get("country"),
            )
            db.merge(race)
            stored += 1
        db.commit()
        return stored
    finally:
        db.close()


def _parse_session_time_ms(time_str: Optional[str]) -> Optional[int]:
    if not time_str:
        return None
    try:
        parts = str(time_str).split(':')
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


def ingest_race_results(season: int, round_num: int) -> int:
    """Fetch results for a specific season/round and persist them. Returns number of results stored."""
    db = _get_db()
    try:
        data = _fetch(f"{JOLPICA_BASE}/{season}/{round_num}/results.json")
        races_data = (
            data.get("MRData", {})
            .get("RaceTable", {})
            .get("Races", [])
        )
        if not races_data:
            return 0

        race_data = races_data[0]
        race = db.query(Race).filter_by(season=season, round=round_num).first()
        if not race:
            race = Race(
                season=season,
                round=round_num,
                race_name=race_data.get("raceName"),
                date=race_data.get("date"),
                circuit_name=race_data.get("Circuit", {}).get("circuitName"),
                country=race_data.get("Circuit", {}).get("Location", {}).get("country"),
            )
            db.add(race)
            db.flush()

        stored = 0
        for result in race_data.get("Results", []):
            driver_data = result.get("Driver", {})
            constructor_data = result.get("Constructor", {})

            driver = db.query(Driver).filter_by(driver_id=driver_data.get("driverId")).first()
            if not driver:
                driver = Driver(
                    driver_id=driver_data.get("driverId"),
                    code=driver_data.get("code"),
                    first_name=driver_data.get("givenName"),
                    last_name=driver_data.get("familyName"),
                    full_name=f"{driver_data.get('givenName', '')} {driver_data.get('familyName', '')}".strip(),
                    nationality=driver_data.get("nationality"),
                    date_of_birth=driver_data.get("dateOfBirth"),
                )
                db.add(driver)
                db.flush()

            constructor = db.query(Constructor).filter_by(constructor_id=constructor_data.get("constructorId")).first()
            if not constructor:
                constructor = Constructor(
                    constructor_id=constructor_data.get("constructorId"),
                    name=constructor_data.get("name"),
                    nationality=constructor_data.get("nationality"),
                )
                db.add(constructor)
                db.flush()

            rr = RaceResult(
                race_id=race.id,
                driver_id=driver.id,
                constructor_id=constructor.id,
                position=int(result.get("position", 0)) if result.get("position") else None,
                position_text=result.get("positionText"),
                points=float(result.get("points", 0)),
                laps=int(result.get("laps", 0)) if result.get("laps") else None,
                time=result.get("Time", {}).get("time") if isinstance(result.get("Time"), dict) else result.get("Time"),
                status=result.get("status"),
                grid=int(result.get("grid", 0)) if result.get("grid") else None,
                fastest_lap=int(result.get("FastestLap", {}).get("lap", 0)) if isinstance(result.get("FastestLap"), dict) else None,
                fastest_lap_time=result.get("FastestLap", {}).get("Time", {}).get("time") if isinstance(result.get("FastestLap"), dict) else None,
            )
            rr.time_ms = _parse_session_time_ms(rr.time)
            db.merge(rr)
            stored += 1
        db.commit()
        return stored
    finally:
        db.close()


def ingest_driver_standings(season: int, round_num: Optional[int] = None) -> int:
    """Fetch driver standings for a season (optionally up to a specific round). Returns count stored."""
    db = _get_db()
    try:
        url = f"{JOLPICA_BASE}/{season}/driverStandings.json"
        if round_num:
            url = f"{JOLPICA_BASE}/{season}/{round_num}/driverStandings.json"
        data = _fetch(url)
        standings_list = (
            data.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [])
        )
        if not standings_list:
            return 0

        standings = standings_list[0].get("DriverStandings", [])
        stored = 0
        for ds in standings:
            driver_data = ds.get("Driver", {})
            driver = db.query(Driver).filter_by(driver_id=driver_data.get("driverId")).first()
            if not driver:
                driver = Driver(
                    driver_id=driver_data.get("driverId"),
                    code=driver_data.get("code"),
                    first_name=driver_data.get("givenName"),
                    last_name=driver_data.get("familyName"),
                    full_name=f"{driver_data.get('givenName', '')} {driver_data.get('familyName', '')}".strip(),
                    nationality=driver_data.get("nationality"),
                    date_of_birth=driver_data.get("dateOfBirth"),
                )
                db.add(driver)
                db.flush()

            standing = DriverStanding(
                season=season,
                round=int(standings_list[0].get("round", 0)) if not round_num else round_num,
                driver_id=driver.id,
                position=int(ds.get("position", 0)) if ds.get("position") else None,
                points=float(ds.get("points", 0)),
                wins=int(ds.get("wins", 0)),
            )
            db.merge(standing)
            stored += 1
        db.commit()
        return stored
    finally:
        db.close()


def ingest_constructor_standings(season: int, round_num: Optional[int] = None) -> int:
    """Fetch constructor standings for a season (optionally up to a specific round). Returns count stored."""
    db = _get_db()
    try:
        url = f"{JOLPICA_BASE}/{season}/constructorStandings.json"
        if round_num:
            url = f"{JOLPICA_BASE}/{season}/{round_num}/constructorStandings.json"
        data = _fetch(url)
        standings_list = (
            data.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [])
        )
        if not standings_list:
            return 0

        standings = standings_list[0].get("ConstructorStandings", [])
        stored = 0
        for cs in standings:
            constructor_data = cs.get("Constructor", {})
            constructor = db.query(Constructor).filter_by(constructor_id=constructor_data.get("constructorId")).first()
            if not constructor:
                constructor = Constructor(
                    constructor_id=constructor_data.get("constructorId"),
                    name=constructor_data.get("name"),
                    nationality=constructor_data.get("nationality"),
                )
                db.add(constructor)
                db.flush()

            standing = ConstructorStanding(
                season=season,
                round=int(standings_list[0].get("round", 0)) if not round_num else round_num,
                constructor_id=constructor.id,
                position=int(cs.get("position", 0)) if cs.get("position") else None,
                points=float(cs.get("points", 0)),
                wins=int(cs.get("wins", 0)),
            )
            db.merge(standing)
            stored += 1
        db.commit()
        return stored
    finally:
        db.close()


def get_season_races(season: int) -> list:
    """Return stored races for a season from the local DB."""
    db = _get_db()
    try:
        return db.query(Race).options(joinedload(Race.results).joinedload(RaceResult.driver), joinedload(Race.results).joinedload(RaceResult.constructor)).filter_by(season=season).order_by(Race.round).all()
    finally:
        db.close()


def get_driver_standings(season: int, round_num: Optional[int] = None) -> list:
    """Return stored driver standings from the local DB."""
    db = _get_db()
    try:
        q = db.query(DriverStanding).filter_by(season=season)
        if round_num:
            q = q.filter_by(round=round_num)
        return q.options(joinedload(DriverStanding.driver)).order_by(DriverStanding.position).all()
    finally:
        db.close()


def get_constructor_standings(season: int, round_num: Optional[int] = None) -> list:
    """Return stored constructor standings from the local DB."""
    db = _get_db()
    try:
        q = db.query(ConstructorStanding).filter_by(season=season)
        if round_num:
            q = q.filter_by(round=round_num)
        return q.options(joinedload(ConstructorStanding.constructor)).order_by(ConstructorStanding.position).all()
    finally:
        db.close()
