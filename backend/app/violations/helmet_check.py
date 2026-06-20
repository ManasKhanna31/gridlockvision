"""
GridlockVision AI — Helmet compliance detection.

Pipeline (fully implemented):
  1. Take a motorcycle/auto-rickshaw detection's bbox from VehicleDetector.
  2. Crop the upper ~40% of the box (head region heuristic).
  3. Run the pluggable helmet classifier on the crop.
  4. Return HELMET / NO_HELMET / UNKNOWN.

UNKNOWN means the classifier model file isn't trained/present yet — see
detection/pluggable_model.py for why this is reported honestly rather
than guessed.

FUTURE ENHANCEMENT: rider head-pose isn't accounted for; a tilted/turned
head can shrink the visible helmet area. A pose-aware crop (e.g. via a
lightweight keypoint model) would improve robustness on real footage.
"""
from dataclasses import dataclass

import numpy as np

from app.core.config import HELMET_MODEL_PATH, HELMET_CONF_THRESHOLD
from app.detection.pluggable_model import PluggableYOLO

_helmet_model = PluggableYOLO(HELMET_MODEL_PATH, task_name="helmet_classifier")


@dataclass
class HelmetResult:
    status: str          # "HELMET" | "NO_HELMET" | "UNKNOWN"
    confidence: float
    stub_mode: bool


def _crop_head_region(frame: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    head_y2 = y1 + max(1, int(h * 0.4))
    x1, y1 = max(0, x1), max(0, y1)
    head_y2 = min(frame.shape[0], head_y2)
    x2 = min(frame.shape[1], x2)
    return frame[y1:head_y2, x1:x2]


def check_helmet(frame: np.ndarray, rider_bbox) -> HelmetResult:
    if not _helmet_model.is_trained:
        return HelmetResult(status="UNKNOWN", confidence=0.0, stub_mode=True)

    crop = _crop_head_region(frame, rider_bbox)
    if crop.size == 0:
        return HelmetResult(status="UNKNOWN", confidence=0.0, stub_mode=False)

    preds = _helmet_model.predict(crop, conf_threshold=HELMET_CONF_THRESHOLD)
    if not preds:
        return HelmetResult(status="UNKNOWN", confidence=0.0, stub_mode=False)

    # take highest-confidence prediction
    label, conf, _ = max(preds, key=lambda p: p[1])
    status = "HELMET" if label.lower() in ("helmet", "with_helmet") else "NO_HELMET"
    return HelmetResult(status=status, confidence=round(conf, 4), stub_mode=False)
