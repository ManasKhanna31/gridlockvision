"""
GridlockVision AI — Seatbelt compliance detection.

Same honest pluggable-model pattern as helmet_check.py. Crops the
driver-side upper torso region of a car/truck/bus detection's bbox and
runs the pluggable seatbelt classifier.

FUTURE ENHANCEMENT: distinguishing "driver seat" from "passenger seat"
visually requires knowing vehicle orientation/driving side; current crop
heuristic assumes a roughly front-on or side-on camera angle typical of
fixed traffic cameras, and is documented as such rather than silently
assumed to be perfect.
"""
from dataclasses import dataclass

import numpy as np

from app.core.config import SEATBELT_MODEL_PATH
from app.detection.pluggable_model import PluggableYOLO

_seatbelt_model = PluggableYOLO(SEATBELT_MODEL_PATH, task_name="seatbelt_classifier")


@dataclass
class SeatbeltResult:
    status: str          # "SEATBELT" | "NO_SEATBELT" | "UNKNOWN"
    confidence: float
    stub_mode: bool


def _crop_driver_region(frame: np.ndarray, vehicle_bbox) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
    w, h = x2 - x1, y2 - y1
    # upper-left quadrant approximation for a front-on camera view
    crop_x2 = x1 + max(1, int(w * 0.55))
    crop_y2 = y1 + max(1, int(h * 0.55))
    x1, y1 = max(0, x1), max(0, y1)
    crop_x2 = min(frame.shape[1], crop_x2)
    crop_y2 = min(frame.shape[0], crop_y2)
    return frame[y1:crop_y2, x1:crop_x2]


def check_seatbelt(frame: np.ndarray, vehicle_bbox) -> SeatbeltResult:
    if not _seatbelt_model.is_trained:
        return SeatbeltResult(status="UNKNOWN", confidence=0.0, stub_mode=True)

    crop = _crop_driver_region(frame, vehicle_bbox)
    if crop.size == 0:
        return SeatbeltResult(status="UNKNOWN", confidence=0.0, stub_mode=False)

    preds = _seatbelt_model.predict(crop)
    if not preds:
        return SeatbeltResult(status="UNKNOWN", confidence=0.0, stub_mode=False)

    label, conf, _ = max(preds, key=lambda p: p[1])
    status = "SEATBELT" if label.lower() in ("seatbelt", "belt") else "NO_SEATBELT"
    return SeatbeltResult(status=status, confidence=round(conf, 4), stub_mode=False)
