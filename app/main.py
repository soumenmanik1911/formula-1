from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import standings, races, features, predict

app = FastAPI(
    title="F1 Prediction Dashboard API",
    description="Phase 1 - serves standings and race data from Jolpica and FastF1.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(standings.router)
app.include_router(races.router)
app.include_router(features.router)
app.include_router(predict.router)



@app.get("/health")
def health():
    return {"status": "ok"}
