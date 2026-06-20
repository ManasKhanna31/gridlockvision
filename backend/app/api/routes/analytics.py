"""
GridlockVision AI — Analytics routes powering the dashboard.
"""
import datetime as dt
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import get_db, Violation
from app.violations.insights_engine import generate_insights, predict_hotspots

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    total = db.query(func.count(Violation.id)).scalar() or 0
    by_type = dict(Counter(
        row[0] for row in db.query(Violation.violation_type).all()
    ))
    by_camera = dict(Counter(
        row[0] for row in db.query(Violation.camera_id).all()
    ))

    top_offenders_rows = (
        db.query(Violation.plate_number, func.count(Violation.id).label("cnt"))
        .filter(Violation.plate_number.isnot(None))
        .group_by(Violation.plate_number)
        .order_by(func.count(Violation.id).desc())
        .limit(10)
        .all()
    )
    top_offenders = [{"plate_number": r[0], "violation_count": r[1]} for r in top_offenders_rows]

    return {
        "total_violations": total,
        "violations_by_type": by_type,
        "violations_by_camera": by_camera,
        "top_offending_vehicles": top_offenders,
    }


@router.get("/trends/daily")
def daily_trends(days: int = 14, db: Session = Depends(get_db)):
    since = dt.datetime.utcnow() - dt.timedelta(days=days)
    rows = db.query(Violation.timestamp).filter(Violation.timestamp >= since).all()
    counts = Counter(r[0].date().isoformat() for r in rows)
    return dict(sorted(counts.items()))


@router.get("/trends/hourly")
def hourly_trends(db: Session = Depends(get_db)):
    rows = db.query(Violation.timestamp).all()
    counts = Counter(r[0].hour for r in rows)
    return {str(h): counts.get(h, 0) for h in range(24)}


@router.get("/heatmap")
def heatmap(db: Session = Depends(get_db)):
    rows = (
        db.query(Violation.gps_lat, Violation.gps_lon, Violation.violation_type)
        .filter(Violation.gps_lat.isnot(None))
        .all()
    )
    return [{"lat": r[0], "lon": r[1], "violation_type": r[2]} for r in rows]


@router.get("/insights")
def insights(lookback_days: int = 7, db: Session = Depends(get_db)):
    return {"insights": generate_insights(db, lookback_days)}


@router.get("/hotspots")
def hotspots(top_n: int = 5, db: Session = Depends(get_db)):
    return {"hotspots": predict_hotspots(db, top_n)}
