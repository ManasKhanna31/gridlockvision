"""
GridlockVision AI — Violation search & retrieval routes.
"""
import datetime as dt
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.models import get_db, Violation
from app.db import crud

router = APIRouter(prefix="/violations", tags=["Violations"])


@router.get("/search")
def search(
    plate_number: Optional[str] = None,
    violation_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
):
    df = dt.datetime.fromisoformat(date_from) if date_from else None
    dtto = dt.datetime.fromisoformat(date_to) if date_to else None

    results = crud.search_violations(db, plate_number, violation_type, df, dtto, limit)
    return [
        {
            "violation_id": v.violation_id,
            "vehicle_type": v.vehicle_type,
            "plate_number": v.plate_number,
            "violation_type": v.violation_type,
            "confidence": v.confidence,
            "timestamp": v.timestamp.isoformat(),
            "camera_id": v.camera_id,
            "gps_lat": v.gps_lat,
            "gps_lon": v.gps_lon,
            "risk_score": v.risk_score,
            "risk_category": v.risk_category,
            "enforcement_priority": v.enforcement_priority,
            "evidence_image_path": v.evidence_image_path,
        }
        for v in results
    ]


@router.get("/{violation_id}")
def get_violation(violation_id: str, db: Session = Depends(get_db)):
    v = db.query(Violation).filter(Violation.violation_id == violation_id).first()
    if not v:
        return {"error": "Not found"}
    return {
        "violation_id": v.violation_id,
        "vehicle_type": v.vehicle_type,
        "plate_number": v.plate_number,
        "violation_type": v.violation_type,
        "confidence": v.confidence,
        "timestamp": v.timestamp.isoformat(),
        "camera_id": v.camera_id,
        "gps_lat": v.gps_lat,
        "gps_lon": v.gps_lon,
        "risk_score": v.risk_score,
        "risk_category": v.risk_category,
        "enforcement_priority": v.enforcement_priority,
        "evidence_image_path": v.evidence_image_path,
        "evidence_json_path": v.evidence_json_path,
        "notes": v.notes,
    }
