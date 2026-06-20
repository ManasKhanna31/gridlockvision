"""
GridlockVision AI — Evidence generation.

For every confirmed violation, produces:
  - An annotated PNG (bounding box, label, plate, timestamp burned in)
  - A JSON metadata sidecar matching the schema specified in the brief

This is the artifact a traffic officer / judge would actually review,
so legibility matters as much as correctness.
"""
import json
import uuid
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from app.core.config import EVIDENCE_DIR


def _draw_label(frame: np.ndarray, bbox, text: str, color=(0, 0, 255)) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def generate_evidence(
    frame: np.ndarray,
    vehicle_bbox: tuple,
    vehicle_type: str,
    plate_number: str | None,
    violation_type: str,
    confidence: float,
    camera_id: str,
    gps_lat: float,
    gps_lon: float,
    plate_bbox: tuple | None = None,
) -> dict:
    """Renders the annotated frame, writes PNG + JSON to disk, and returns
    the metadata dict (also what gets persisted to the DB / returned by API).
    """
    violation_id = str(uuid.uuid4())
    timestamp = dt.datetime.utcnow().isoformat() + "Z"

    annotated = frame.copy()
    label = f"{vehicle_type.upper()} | {violation_type} | {confidence:.2f}"
    _draw_label(annotated, vehicle_bbox, label)

    if plate_bbox is not None:
        _draw_label(annotated, plate_bbox, plate_number or "UNREADABLE", color=(255, 140, 0))

    banner = f"{camera_id}  |  {timestamp}  |  GPS {gps_lat:.4f},{gps_lon:.4f}"
    cv2.rectangle(annotated, (0, annotated.shape[0] - 28), (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
    cv2.putText(annotated, banner, (8, annotated.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    image_filename = f"{violation_id}.png"
    json_filename = f"{violation_id}.json"
    image_path = EVIDENCE_DIR / image_filename
    json_path = EVIDENCE_DIR / json_filename

    cv2.imwrite(str(image_path), annotated)

    metadata = {
        "violation_id": violation_id,
        "timestamp": timestamp,
        "camera_id": camera_id,
        "gps_coordinates": {"lat": gps_lat, "lon": gps_lon},
        "vehicle_type": vehicle_type,
        "plate_number": plate_number,
        "violation_type": violation_type,
        "confidence": confidence,
        "evidence_image_path": str(image_path),
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    metadata["evidence_json_path"] = str(json_path)
    return metadata
