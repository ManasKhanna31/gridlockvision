"""
GridlockVision AI — License plate recognition.

IMPLEMENTED:
  - Plate localization: pluggable YOLO model (same honest stub pattern as
    helmet/seatbelt). If untrained, falls back to a classical CV plate
    locator using contour + aspect-ratio heuristics on edge-detected
    regions within the vehicle bbox — genuinely functional for clean,
    front-on, well-lit plates, with real (lower) accuracy than a trained
    detector, which is disclosed via `localization_method`.
  - Skew correction: minAreaRect + affine warp to deskew the plate crop
    before OCR — real, always-on classical CV step.
  - OCR: EasyOCR (works offline after first model download, GPU-optional).
  - Indian plate format validation: regex check against the standard
    "SS DD LL DDDD" pattern (e.g. DL 01 AB 1234), used only to assign a
    confidence boost / format_valid flag — never to reject/replace the
    raw OCR text, so OCR output is never silently altered.

FUTURE ENHANCEMENT: a trained plate detector (vs the classical fallback)
substantially improves localization recall on tilted, dirty, or partially
occluded plates — recommended first thing to fine-tune given a labelled
dataset, since OCR quality is bottlenecked by crop quality.
"""
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.core.config import PLATE_MODEL_PATH, PLATE_CONF_THRESHOLD, OCR_MIN_CONF
from app.detection.pluggable_model import PluggableYOLO

_plate_model = PluggableYOLO(PLATE_MODEL_PATH, task_name="plate_detector")

_INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4}$")

_easyocr_reader = None


def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


@dataclass
class PlateResult:
    plate_number: Optional[str]
    plate_bbox: Optional[tuple]
    ocr_confidence: float
    format_valid: bool
    localization_method: str  # "trained_model" | "classical_cv_fallback" | "not_found"


def _locate_plate_classical(vehicle_crop: np.ndarray) -> Optional[tuple]:
    """Edge + contour based plate locator. Looks for rectangular regions
    with a plate-like aspect ratio (~2:1 to ~5:1) inside the lower half
    of the vehicle crop (plates are mounted low on Indian vehicles).
    """
    h, w = vehicle_crop.shape[:2]
    lower_half = vehicle_crop[int(h * 0.4):, :]
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_score = 0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area = cw * ch
        if 2.0 <= aspect <= 5.5 and area > (w * h * 0.01):
            if area > best_score:
                best_score = area
                # map back to full crop coordinates (offset by lower_half start)
                y_offset = int(h * 0.4)
                best_box = (x, y + y_offset, x + cw, y + y_offset + ch)

    return best_box


def _deskew_plate(plate_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return plate_crop

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1.0:
        return plate_crop  # not worth warping

    (h, w) = plate_crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(plate_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _clean_ocr_text(raw: str) -> str:
    text = raw.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def recognize_plate(frame: np.ndarray, vehicle_bbox: tuple) -> PlateResult:
    x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    vehicle_crop = frame[y1:y2, x1:x2]

    if vehicle_crop.size == 0:
        return PlateResult(None, None, 0.0, False, "not_found")

    plate_box_local = None
    method = "not_found"

    if _plate_model.is_trained:
        preds = _plate_model.predict(vehicle_crop, conf_threshold=PLATE_CONF_THRESHOLD)
        if preds:
            _, _, plate_box_local = max(preds, key=lambda p: p[1])
            method = "trained_model"

    if plate_box_local is None:
        plate_box_local = _locate_plate_classical(vehicle_crop)
        if plate_box_local is not None:
            method = "classical_cv_fallback"

    if plate_box_local is None:
        return PlateResult(None, None, 0.0, False, "not_found")

    px1, py1, px2, py2 = [int(v) for v in plate_box_local]
    plate_crop = vehicle_crop[max(0, py1):py2, max(0, px1):px2]
    if plate_crop.size == 0:
        return PlateResult(None, None, 0.0, False, method)

    plate_crop = _deskew_plate(plate_crop)
    # upscale small crops — meaningfully helps OCR on distant plates
    if plate_crop.shape[0] < 60:
        scale = 60 / plate_crop.shape[0]
        plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    reader = _get_ocr_reader()
    results = reader.readtext(plate_crop)

    if not results:
        return PlateResult(None, (x1 + px1, y1 + py1, x1 + px2, y1 + py2), 0.0, False, method)

    # concatenate all detected text fragments (plates sometimes split into
    # two OCR boxes e.g. state-code line vs number line)
    fragments = sorted(results, key=lambda r: r[0][0][0])  # left-to-right by x
    best_conf = max(r[2] for r in fragments)
    combined_text = "".join(_clean_ocr_text(r[1]) for r in fragments)

    if best_conf < OCR_MIN_CONF:
        return PlateResult(None, (x1 + px1, y1 + py1, x1 + px2, y1 + py2), round(best_conf, 4), False, method)

    format_valid = bool(_INDIAN_PLATE_REGEX.match(combined_text))

    return PlateResult(
        plate_number=combined_text,
        plate_bbox=(x1 + px1, y1 + py1, x1 + px2, y1 + py2),
        ocr_confidence=round(best_conf, 4),
        format_valid=format_valid,
        localization_method=method,
    )
