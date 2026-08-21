# F1 Prediction Dashboard — Phase 1 Build Prompt (FastAPI + Data Pipeline)

Copy everything below into Claude Code (or paste it as a task to any AI coding agent). It's written so the agent works through it in order, one step at a time, and doesn't skip ahead.

---

## ROLE

You are a senior Python backend engineer. Build the first phase of an F1 prediction dashboard: a clean Python environment, a FastAPI backend skeleton, and a working data pipeline that pulls real F1 data. Do not build the frontend or the ML model yet — this phase stops once the API can serve real F1 data from a local database.

## CONTEXT

- Data source for historical results/standings: **Jolpica-F1** API (Ergast-compatible successor), base URL `https://api.jolpi.ca/ergast/f1/`. No API key needed. Free tier is rate-limited to 200 req/hour, so cache responses locally instead of re-fetching.
- Data source for lap/telemetry data: **FastF1** Python library (`pip install fastf1`), pulls official F1 timing data from 2018 onward.
- Storage: SQLite for now (simple, file-based, zero setup cost). We'll move to Postgres later if needed.
- End goal of this phase: `GET /standings/drivers`, `GET /standings/constructors`, and `GET /races/{season}` return real data pulled from the two sources above, stored locally, served through FastAPI.

## STEP 1 — Environment setup

Create an isolated Python virtual environment for this project so it doesn't touch global packages.

```bash
python3 -m venv f1-dashboard-env
source f1-dashboard-env/bin/activate   # on Windows: f1-dashboard-env\Scripts\activate
python -m pip install --upgrade pip
```

Confirm the venv is active (prompt should show `(f1-dashboard-env)`) before continuing to any install step.

## STEP 2 — Project structure

Create this folder layout:

```
f1-dashboard/
├── f1-dashboard-env/        # venv, already created — add to .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entrypoint
│   ├── database.py          # SQLite connection/session setup
│   ├── models.py            # SQLAlchemy table models
│   ├── routers/
│   │   ├── standings.py
│   │   └── races.py
│   └── ingestion/
│       ├── jolpica_client.py    # pulls results/standings from Jolpica
│       └── fastf1_client.py     # pulls lap/telemetry via FastF1
├── requirements.txt
├── .gitignore
└── README.md
```

## STEP 3 — Install dependencies

```bash
pip install fastapi uvicorn[standard] sqlalchemy requests fastf1 pandas python-dotenv
pip freeze > requirements.txt
```

## STEP 4 — Database models

In `app/models.py`, define SQLAlchemy models for: `Driver`, `Constructor`, `Race`, `RaceResult`, `DriverStanding`, `ConstructorStanding`. Keep fields minimal for now — id, name, nationality, points, position, season, round. We'll expand later once the ML feature list is locked in.

## STEP 5 — Jolpica ingestion client

In `app/ingestion/jolpica_client.py`, write functions that:
- Fetch a season's full race schedule
- Fetch race results for a given season/round
- Fetch current driver standings and constructor standings
- Save all of it into the SQLite tables from Step 4
- Cache responses to disk (or check "already in DB" before re-fetching) so we respect the 200 req/hour limit

## STEP 6 — FastF1 ingestion client

In `app/ingestion/fastf1_client.py`, write a function that pulls lap times and basic session data for a given season/round using FastF1's `fastf1.get_session()`. Enable FastF1's built-in cache (`fastf1.Cache.enable_cache('cache/')`) so repeat runs don't re-download.

## STEP 7 — FastAPI routes

Wire up `app/main.py` with the routers. Endpoints to build:
- `GET /standings/drivers?season=2026`
- `GET /standings/constructors?season=2026`
- `GET /races/{season}` — list of races with results if available

## STEP 8 — Run and verify

```bash
uvicorn app.main:app --reload
```

Hit `http://localhost:8000/docs` to confirm the auto-generated Swagger UI shows all three endpoints, and that they return real data (not empty arrays) after running the ingestion functions once manually.

## STOP HERE

Once all three endpoints return real 2026-season data pulled from Jolpica and stored in SQLite, this phase is done. Do not start on the ML model or frontend — report back what's working and what data looks off, so we can lock the schema before building features on top of it.
