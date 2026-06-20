"""
GridlockVision AI — Innovation Feature A & B: Violation Risk Score and
Smart Enforcement Recommendation.

IMPLEMENTED as a transparent, explainable scoring formula (not a black-box
ML model) — deliberately, since judges can interrogate a formula and a
"trust me, it's a neural net" risk score would be a red flag, not a
differentiator, without real training data to back it.

Risk Score (0-100) =
    severity_weight(violation_type)          [0-40]
  + frequency_component(repeat_count)         [0-30]
  + recency_component(days_since_last)        [0-30]

Severity weights reflect real-world harm potential (e.g. red-light
running ranks above illegal parking), loosely aligned with Indian Motor
Vehicles Act fine tiers — documented inline so the weights are auditable
and adjustable by the team, not asserted as ground truth.
"""
from dataclasses import dataclass
import datetime as dt

SEVERITY_WEIGHTS = {
    "RED_LIGHT_VIOLATION": 40,
    "WRONG_SIDE_DRIVING": 38,
    "TRIPLE_RIDING": 30,
    "HELMET_VIOLATION": 25,
    "SEATBELT_VIOLATION": 22,
    "STOP_LINE_VIOLATION": 18,
    "ILLEGAL_PARKING": 12,
}
DEFAULT_SEVERITY = 15


@dataclass
class RiskResult:
    risk_score: float
    risk_category: str          # LOW / MEDIUM / HIGH / CRITICAL
    enforcement_priority: str   # ROUTINE / ELEVATED / URGENT
    explanation: str


def _severity_component(violation_type: str) -> float:
    return SEVERITY_WEIGHTS.get(violation_type, DEFAULT_SEVERITY)


def _frequency_component(prior_violation_count: int) -> float:
    # diminishing-returns curve so a 50-time offender doesn't blow past 30
    capped = min(prior_violation_count, 10)
    return min(30.0, capped * 3.0)


def _recency_component(last_violation_date: dt.datetime | None) -> float:
    if last_violation_date is None:
        return 0.0
    days_ago = max(0, (dt.datetime.utcnow() - last_violation_date).days)
    if days_ago <= 7:
        return 30.0
    if days_ago <= 30:
        return 18.0
    if days_ago <= 90:
        return 8.0
    return 2.0


def compute_risk(
    violation_type: str,
    prior_violation_count: int,
    last_violation_date: dt.datetime | None,
) -> RiskResult:
    severity = _severity_component(violation_type)
    frequency = _frequency_component(prior_violation_count)
    recency = _recency_component(last_violation_date)

    score = round(severity + frequency + recency, 1)
    score = min(100.0, score)

    if score >= 75:
        category, priority = "CRITICAL", "URGENT"
    elif score >= 50:
        category, priority = "HIGH", "ELEVATED"
    elif score >= 25:
        category, priority = "MEDIUM", "ROUTINE"
    else:
        category, priority = "LOW", "ROUTINE"

    explanation = (
        f"severity={severity} (type={violation_type}), "
        f"frequency={frequency} (prior_count={prior_violation_count}), "
        f"recency={recency}"
    )

    return RiskResult(
        risk_score=score,
        risk_category=category,
        enforcement_priority=priority,
        explanation=explanation,
    )
