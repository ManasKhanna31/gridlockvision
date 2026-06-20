"""
GridlockVision AI — FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs available at /docs once running.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import API_TITLE, API_VERSION, CORS_ORIGINS, EVIDENCE_DIR
from app.db.models import init_db
from app.api.routes import demo, violations, analytics

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Automated Photo Identification and Classification for Traffic "
        "Violations Using Computer Vision — Flipkart Gridlock Hackathon 2.0, "
        "Theme 3."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

app.include_router(demo.router)
app.include_router(violations.router)
app.include_router(analytics.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {
        "project": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
