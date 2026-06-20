"""
GridlockVision AI — Triple riding detection.

IMPLEMENTED: Real working logic, no model training needed.
  1. For each detected motorcycle bbox, count how many "pedestrian"
     (person-class) detections have their lower-body point (bottom-center
     of their bbox) falling inside the motorcycle's bbox, expanded
     slightly to tolerate pose variance.
  2. If rider_count > TRIPLE_RIDING_MAX_RIDERS -> flag.

This is a clean geometric heuristic — robust for the common case where
riders are seated in a vertical line on the bike, which is the visual
signature of triple riding in real footage.

FUTURE ENHANCEMENT: pose-estimation (e.g. lightweight keypoint model)
would let us count riders even when partially occluded by each other,
which a pure bbox-containment check can undercount.
"""
from dataclasses import dataclass

from app.core.config import TRIPLE_RIDING_MAX_RIDERS


@dataclass
class TripleRidingResult:
    rider_count: int
    is_violation: bool


def _bbox_contains_point(bbox, px, py, expand_ratio: float = 0.15) -> bool:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    ex1 = x1 - w * expand_ratio
    ex2 = x2 + w * expand_ratio
    ey1 = y1 - h * expand_ratio
    ey2 = y2 + h * expand_ratio
    return ex1 <= px <= ex2 and ey1 <= py <= ey2


def count_riders(motorcycle_bbox, person_detections: list) -> TripleRidingResult:
    """person_detections: list of Detection objects (label == 'pedestrian')."""
    count = 0
    for p in person_detections:
        x1, y1, x2, y2 = p.bbox
        # use the person's vertical-center point — more robust than feet,
        # since lower legs are frequently occluded by the bike body
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if _bbox_contains_point(motorcycle_bbox, cx, cy):
            count += 1

    return TripleRidingResult(
        rider_count=count,
        is_violation=count > TRIPLE_RIDING_MAX_RIDERS,
    )
