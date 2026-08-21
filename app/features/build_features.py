import os
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import SessionLocal, engine, Base
from app.models import Driver, Constructor, Race, RaceResult, DriverRaceFeature

Base.metadata.create_all(bind=engine)


def _get_db() -> Session:
    return SessionLocal()


def compute_rolling_stats(driver_id: int, before_race_id: int, window: int = 5) -> Tuple[Optional[float], Optional[float]]:
    db = _get_db()
    try:
        target_race = db.query(Race).get(before_race_id)
        if not target_race:
            return None, None
        results = (
            db.query(RaceResult)
            .join(Race)
            .filter(RaceResult.driver_id == driver_id)
            .filter(Race.date < target_race.date)
            .order_by(Race.date.asc(), Race.round.asc())
            .all()
        )
        if not results:
            return None, None
        recent = results[-window:]
        finishes = [r.position for r in recent if r.position is not None]
        points = [r.points for r in recent if r.points is not None]
        avg_finish = sum(finishes) / len(finishes) if finishes else None
        avg_points = sum(points) / len(points) if points else None
        return avg_finish, avg_points
    finally:
        db.close()


def compute_constructor_rolling_stats(constructor_id: int, before_race_id: int, window: int = 5) -> Optional[float]:
    db = _get_db()
    try:
        target_race = db.query(Race).get(before_race_id)
        if not target_race:
            return None
        results = (
            db.query(RaceResult)
            .join(Race)
            .filter(RaceResult.constructor_id == constructor_id)
            .filter(Race.date < target_race.date)
            .order_by(Race.date.asc(), Race.round.asc())
            .all()
        )
        if not results:
            return None
        recent = results[-window:]
        points = [r.points for r in recent if r.points is not None]
        return sum(points) / len(points) if points else None
    finally:
        db.close()


def update_elo_ratings(season: int) -> dict:
    db = _get_db()
    try:
        races = (
            db.query(Race)
            .filter(Race.season == season)
            .filter(Race.date.isnot(None))
            .order_by(Race.date.asc(), Race.round.asc())
            .all()
        )
        drivers = db.query(Driver).all()
        elo = {d.id: 1500.0 for d in drivers}
        pre_race_elo = {d.id: {} for d in drivers}
        for race in races:
            for driver_id in elo:
                pre_race_elo[driver_id][race.id] = elo[driver_id]
            results = (
                db.query(RaceResult)
                .filter(RaceResult.race_id == race.id)
                .filter(RaceResult.position.isnot(None))
                .order_by(RaceResult.position.asc())
                .all()
            )
            if len(results) < 2:
                continue
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    ra = elo[results[i].driver_id]
                    rb = elo[results[j].driver_id]
                    expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
                    expected_b = 1.0 - expected_a
                    actual_a = 1.0
                    actual_b = 0.0
                    elo[results[i].driver_id] = ra + 32.0 * (actual_a - expected_a)
                    elo[results[j].driver_id] = rb + 32.0 * (actual_b - expected_b)
        return pre_race_elo
    finally:
        db.close()


def compute_circuit_history(driver_id: int, circuit_name: str, before_race_id: int) -> Optional[float]:
    db = _get_db()
    try:
        target_race = db.query(Race).get(before_race_id)
        if not target_race:
            return None
        results = (
            db.query(RaceResult)
            .join(Race)
            .filter(RaceResult.driver_id == driver_id)
            .filter(Race.circuit_name == circuit_name)
            .filter(Race.date < target_race.date)
            .filter(RaceResult.position.isnot(None))
            .all()
        )
        if not results:
            return None
        positions = [r.position for r in results]
        return sum(positions) / len(positions)
    finally:
        db.close()


def compute_points_before_race(driver_id: int, race_id: int) -> float:
    db = _get_db()
    try:
        target_race = db.query(Race).get(race_id)
        if not target_race:
            return 0.0
        total = (
            db.query(func.sum(RaceResult.points))
            .join(Race)
            .filter(RaceResult.driver_id == driver_id)
            .filter(Race.season == target_race.season)
            .filter(Race.date < target_race.date)
            .scalar()
        )
        return float(total) if total is not None else 0.0
    finally:
        db.close()


def compute_standing_position_before_race(driver_id: int, race_id: int) -> Optional[int]:
    db = _get_db()
    try:
        target_race = db.query(Race).get(race_id)
        if not target_race:
            return None
        all_points = (
            db.query(RaceResult.driver_id, func.sum(RaceResult.points).label("total"))
            .join(Race)
            .filter(Race.season == target_race.season)
            .filter(Race.date < target_race.date)
            .group_by(RaceResult.driver_id)
            .all()
        )
        if not all_points:
            return None
        all_points_sorted = sorted(all_points, key=lambda x: (-(x[1] or 0), x[0]))
        for rank, (d_id, _) in enumerate(all_points_sorted, 1):
            if d_id == driver_id:
                return rank
        return None
    finally:
        db.close()


def build_all_features() -> int:
    db = _get_db()
    try:
        db.query(DriverRaceFeature).delete()
        db.commit()
        races = (
            db.query(Race)
            .filter(Race.date.isnot(None))
            .order_by(Race.date.asc(), Race.round.asc())
            .all()
        )
        if not races:
            return 0
        seasons = sorted(set(r.season for r in races))
        all_pre_race_elo = {}
        for season in seasons:
            season_elo = update_elo_ratings(season)
            for driver_id, race_elos in season_elo.items():
                all_pre_race_elo.setdefault(driver_id, {}).update(race_elos)
        count = 0
        for race in races:
            results = db.query(RaceResult).filter(RaceResult.race_id == race.id).all()
            for rr in results:
                driver = db.query(Driver).get(rr.driver_id)
                constructor = db.query(Constructor).get(rr.constructor_id)
                if not driver or not constructor:
                    continue
                qualifying_position = rr.grid
                avg_finish, avg_points = compute_rolling_stats(rr.driver_id, race.id)
                const_avg_points = compute_constructor_rolling_stats(rr.constructor_id, race.id)
                driver_elo = all_pre_race_elo.get(rr.driver_id, {}).get(race.id, 1500.0)
                circuit_avg = compute_circuit_history(rr.driver_id, race.circuit_name, race.id)
                points_before = compute_points_before_race(rr.driver_id, race.id)
                standing_pos = compute_standing_position_before_race(rr.driver_id, race.id)
                feature = DriverRaceFeature(
                    driver_id=rr.driver_id,
                    race_id=race.id,
                    season=race.season,
                    round=race.round,
                    qualifying_position=qualifying_position,
                    rolling_avg_finish_last5=avg_finish,
                    rolling_avg_points_last5=avg_points,
                    constructor_rolling_avg_points_last5=const_avg_points,
                    driver_elo=driver_elo,
                    circuit_avg_finish=circuit_avg,
                    points_before_race=points_before,
                    standing_position_before_race=standing_pos,
                    finish_position=rr.position,
                )
                db.add(feature)
                count += 1
        db.commit()
        return count
    finally:
        db.close()
