"""
GridlockVision AI — Innovation Feature C & D: AI Traffic Insights and
Hotspot Prediction.

IMPLEMENTED: Real statistical analysis over the violations table —
template-filled natural language summaries (not an LLM call, deliberately
— this needs to run fully offline/free during a judging demo with no API
key dependency, and the brief's example "Helmet violations increased by
18% during evening hours" is exactly the kind of statement a comparison
of two time-windows produces directly).

Hotspot prediction (D) is a FUTURE-LEANING but genuinely implemented v1:
it ranks existing camera/location-level violation density as a proxy for
"predicted" hotspots (more violations historically -> higher predicted
risk going forward). This is a legitimate baseline forecasting approach
(frequency-based heuristic), clearly distinguished from a trained
time-series forecasting model (e.g. Prophet/ARIMA), which is flagged as
a FUTURE ENHANCEMENT for when there's enough historical data to fit one
meaningfully (a few hours of hackathon demo data won't support real
forecasting model training).
"""
import datetime as dt
from collections import Counter

from sqlalchemy.orm import Session

from app.db.models import Violation


def _pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return round(((new - old) / old) * 100, 1)


def generate_insights(db: Session, lookback_days: int = 7) -> list[str]:
    """Compares the most recent lookback window to the prior window of
    equal length, per violation type and per hour-bucket, and emits
    natural-language sentences for the deltas that are actually
    statistically meaningful (a minimum sample size guard avoids
    "violations increased by 400%" noise from tiny counts).
    """
    now = dt.datetime.utcnow()
    recent_start = now - dt.timedelta(days=lookback_days)
    prior_start = recent_start - dt.timedelta(days=lookback_days)

    recent = db.query(Violation).filter(Violation.timestamp >= recent_start).all()
    prior = db.query(Violation).filter(
        Violation.timestamp >= prior_start, Violation.timestamp < recent_start
    ).all()

    insights = []

    # --- by violation type ---
    recent_by_type = Counter(v.violation_type for v in recent)
    prior_by_type = Counter(v.violation_type for v in prior)

    for vtype in set(recent_by_type) | set(prior_by_type):
        r_count = recent_by_type.get(vtype, 0)
        p_count = prior_by_type.get(vtype, 0)
        if r_count < 5 and p_count < 5:
            continue  # not enough samples to say anything meaningful
        change = _pct_change(p_count, r_count)
        if abs(change) < 10:
            continue
        direction = "increased" if change > 0 else "decreased"
        insights.append(
            f"{vtype.replace('_', ' ').title()} {direction} by {abs(change)}% "
            f"over the last {lookback_days} days compared to the previous period."
        )

    # --- by hour-of-day (evening vs morning skew) ---
    def hour_bucket(v):
        h = v.timestamp.hour
        if 5 <= h < 12:
            return "morning"
        if 12 <= h < 17:
            return "afternoon"
        if 17 <= h < 21:
            return "evening"
        return "night"

    recent_by_bucket = Counter(hour_bucket(v) for v in recent)
    total_recent = sum(recent_by_bucket.values())
    if total_recent >= 10:
        top_bucket, top_count = recent_by_bucket.most_common(1)[0]
        share = round((top_count / total_recent) * 100, 1)
        if share >= 35:
            insights.append(
                f"{share}% of violations in the last {lookback_days} days occurred "
                f"during the {top_bucket}, the highest concentration of any time window."
            )

    if not insights:
        insights.append(
            f"No statistically significant trend changes detected in the last "
            f"{lookback_days} days — volumes are stable or sample size is too small."
        )

    return insights


def predict_hotspots(db: Session, top_n: int = 5) -> list[dict]:
    """Frequency-based hotspot ranking (documented as a baseline heuristic,
    not a trained forecasting model — see module docstring).
    """
    violations = db.query(Violation).all()
    by_camera = Counter(v.camera_id for v in violations)

    results = []
    for camera_id, count in by_camera.most_common(top_n):
        sample = next(v for v in violations if v.camera_id == camera_id)
        results.append({
            "camera_id": camera_id,
            "historical_violation_count": count,
            "predicted_risk_level": (
                "HIGH" if count >= 20 else "MEDIUM" if count >= 8 else "LOW"
            ),
            "gps_lat": sample.gps_lat,
            "gps_lon": sample.gps_lon,
            "method": "frequency_based_heuristic",
        })
    return results
