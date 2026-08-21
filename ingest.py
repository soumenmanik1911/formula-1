import sys
from app.ingestion import jolpica_client

SEASON = 2026

if __name__ == "__main__":
    print(f"Ingesting season {SEASON} schedule...")
    races = jolpica_client.ingest_season_schedule(SEASON)
    print(f"Stored {races} races")

    print("Fetching driver standings...")
    ds = jolpica_client.ingest_driver_standings(SEASON)
    print(f"Stored {ds} driver standings")

    print("Fetching constructor standings...")
    cs = jolpica_client.ingest_constructor_standings(SEASON)
    print(f"Stored {cs} constructor standings")

    print("Done.")
