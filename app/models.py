from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(String, unique=True, index=True, nullable=False)
    code = Column(String, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    nationality = Column(String)
    date_of_birth = Column(String)


class Constructor(Base):
    __tablename__ = "constructors"

    id = Column(Integer, primary_key=True, index=True)
    constructor_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    nationality = Column(String)


class Race(Base):
    __tablename__ = "races"

    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, index=True, nullable=False)
    round = Column(Integer, index=True, nullable=False)
    race_name = Column(String, nullable=False)
    date = Column(String)
    circuit_name = Column(String)
    country = Column(String)
    results = relationship("RaceResult", back_populates="race")
    lap_data = relationship("LapData", back_populates="race")
    __table_args__ = (UniqueConstraint("season", "round", name="uix_season_round"),)


class RaceResult(Base):
    __tablename__ = "race_results"

    id = Column(Integer, primary_key=True, index=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    constructor_id = Column(Integer, ForeignKey("constructors.id"), nullable=False)
    position = Column(Integer)
    position_text = Column(String)
    points = Column(Float, default=0.0)
    laps = Column(Integer)
    time = Column(String)
    time_ms = Column(Integer)
    status = Column(String)
    grid = Column(Integer)
    fastest_lap = Column(Integer)
    fastest_lap_time = Column(String)

    race = relationship("Race", back_populates="results")
    driver = relationship("Driver")
    constructor = relationship("Constructor")


class DriverStanding(Base):
    __tablename__ = "driver_standings"

    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, index=True, nullable=False)
    round = Column(Integer, index=True, nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    position = Column(Integer)
    points = Column(Float, default=0.0)
    wins = Column(Integer, default=0)

    driver = relationship("Driver")


class ConstructorStanding(Base):
    __tablename__ = "constructor_standings"

    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, index=True, nullable=False)
    round = Column(Integer, index=True, nullable=False)
    constructor_id = Column(Integer, ForeignKey("constructors.id"), nullable=False)
    position = Column(Integer)
    points = Column(Float, default=0.0)
    wins = Column(Integer, default=0)

    constructor = relationship("Constructor")


class LapData(Base):
    __tablename__ = "lap_data"

    id = Column(Integer, primary_key=True, index=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    driver_code = Column(String, index=True, nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time = Column(String)
    lap_time_ms = Column(Integer)
    tyre_compound = Column(String)
    tyre_age = Column(Integer)
    stint = Column(Integer)
    fresh_tyre = Column(String)
    track_status = Column(String)
    lap_start_time = Column(String)
    driver_number = Column(Integer)
    team = Column(String)

    race = relationship("Race", back_populates="lap_data")


# One row per driver per completed race for model training.
# finish_position is the TARGET label, not an input feature.
class DriverRaceFeature(Base):
    __tablename__ = "driver_race_features"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    season = Column(Integer, index=True, nullable=False)
    round = Column(Integer, index=True, nullable=False)
    qualifying_position = Column(Integer)  # mapped from RaceResult.grid
    rolling_avg_finish_last5 = Column(Float)
    rolling_avg_points_last5 = Column(Float)
    constructor_rolling_avg_points_last5 = Column(Float)
    driver_elo = Column(Float)
    circuit_avg_finish = Column(Float)
    points_before_race = Column(Float)
    standing_position_before_race = Column(Integer)
    finish_position = Column(Integer)  # TARGET / LABEL

    __table_args__ = (UniqueConstraint("driver_id", "race_id", name="uix_driver_race"),)
