"""
GridlockVision AI — Pipeline orchestrator.

This is the spine of Demo Mode: image/frame in -> violations out, with
every intermediate step exposed in the returned dict so the dashboard
can render "Step 1: detect vehicles", "Step 2: detect violations", etc.
live, exactly as the brief's Judge Demo Mode requires.
"""
import time
import datetime as dt
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_CAMERA_ID, DEFAULT_GPS_LAT, DEFAULT_GPS_LON
from app.preprocessing.pipeline import auto_preprocess
from app.detection.vehicle_detector import VehicleDetector, Detection
from app.violations.helmet_check import check_helmet
from app.violations.seatbelt_check import check_seatbelt
from app.violations.triple_riding import count_riders
from app.violations.trajectory_rules import (
    TrackHistory, check_wrong_side, check_stop_line, check_red_light_violation,
    check_illegal_parking, read_signal_state,
)
from app.violations.roi_config import (
    DEMO_CAMERA_CONFIG, denormalize_line, denormalize_polygon, denormalize_box,
)
from app.ocr.plate_recognition import recognize_plate
from app.violations.evidence_generator import generate_evidence
from app.violations.risk_engine import compute_risk
from app.db import crud

_detector = VehicleDetector()
_track_history = TrackHistory()


def _classify_vehicle_group(detections: list[Detection]):
    motorcycles = [d for d in detections if d.label in ("motorcycle", "auto_rickshaw")]
    cars_trucks_buses = [d for d in detections if d.label in ("car", "truck", "bus")]
    pedestrians = [d for d in detections if d.label == "pedestrian"]
    return motorcycles, cars_trucks_buses, pedestrians


def process_frame(
    frame: np.ndarray,
    db: Session,
    camera_id: str = DEFAULT_CAMERA_ID,
    gps_lat: float = DEFAULT_GPS_LAT,
    gps_lon: float = DEFAULT_GPS_LON,
    is_video_frame: bool = False,
    frame_ts: Optional[float] = None,
) -> dict:
    """Runs the full pipeline on a single frame and returns a structured
    result with every intermediate step, plus any violations persisted
    to the DB with generated evidence.

    Every stage is timed with time.perf_counter() and returned under
    `timing_ms` — real, measured latency on whatever hardware runs this
    process, not an estimate.
    """
    steps = {}
    timing_ms = {}
    now_ts = frame_ts if frame_ts is not None else time.time()
    h, w = frame.shape[:2]

    # Step 0: preprocessing
    _t0 = time.perf_counter()
    processed_frame, preprocess_info = auto_preprocess(frame)
    timing_ms["1_preprocessing"] = round((time.perf_counter() - _t0) * 1000, 2)
    steps["1_preprocessing"] = preprocess_info

    # Step 1: detect + track vehicles/people
    _t0 = time.perf_counter()
    detections = _detector.detect_and_track(processed_frame, persist=is_video_frame)
    timing_ms["2_detection"] = round((time.perf_counter() - _t0) * 1000, 2)
    steps["2_detection"] = [
        {"label": d.label, "confidence": d.confidence, "bbox": d.bbox, "track_id": d.track_id,
         "approximated": d.vehicle_type_is_approximated}
        for d in detections
    ]

    motorcycles, four_wheelers, pedestrians = _classify_vehicle_group(detections)

    # update track history for trajectory-based rules
    for d in detections:
        if d.track_id is not None:
            x1, y1, x2, y2 = d.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            _track_history.update(d.track_id, center, now_ts)

    stop_line_px = denormalize_line(DEMO_CAMERA_CONFIG.stop_line, w, h)
    no_parking_px = denormalize_polygon(DEMO_CAMERA_CONFIG.no_parking_zone, w, h)
    signal_box_px = denormalize_box(DEMO_CAMERA_CONFIG.signal_roi, w, h)
    signal_state = read_signal_state(processed_frame, signal_box_px)
    steps["3_signal_state"] = signal_state

    violations_found = []

    # --- Helmet + triple riding (motorcycle-based) ---
    for moto in motorcycles:
        helmet_result = check_helmet(processed_frame, moto.bbox)
        if helmet_result.status == "NO_HELMET":
            violations_found.append({
                "violation_type": "HELMET_VIOLATION",
                "vehicle_bbox": moto.bbox,
                "vehicle_type": moto.label,
                "confidence": helmet_result.confidence,
                "stub_mode": helmet_result.stub_mode,
            })

        riding_result = count_riders(moto.bbox, pedestrians)
        if riding_result.is_violation:
            violations_found.append({
                "violation_type": "TRIPLE_RIDING",
                "vehicle_bbox": moto.bbox,
                "vehicle_type": moto.label,
                "confidence": moto.confidence,
                "rider_count": riding_result.rider_count,
                "stub_mode": False,
            })

        if moto.track_id is not None:
            ws = check_wrong_side(moto.track_id, _track_history, DEMO_CAMERA_CONFIG.lane_direction)
            if ws.is_violation:
                violations_found.append({
                    "violation_type": "WRONG_SIDE_DRIVING",
                    "vehicle_bbox": moto.bbox,
                    "vehicle_type": moto.label,
                    "confidence": 0.7,
                    "angle_deviation": ws.angle_deviation_deg,
                    "stub_mode": False,
                })

    # --- Seatbelt, stop-line, red-light, illegal parking (4-wheelers) ---
    for veh in four_wheelers:
        seatbelt_result = check_seatbelt(processed_frame, veh.bbox)
        if seatbelt_result.status == "NO_SEATBELT":
            violations_found.append({
                "violation_type": "SEATBELT_VIOLATION",
                "vehicle_bbox": veh.bbox,
                "vehicle_type": veh.label,
                "confidence": seatbelt_result.confidence,
                "stub_mode": seatbelt_result.stub_mode,
            })

        if veh.track_id is not None:
            x1, y1, x2, y2 = veh.bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            sl_result = check_stop_line(veh.track_id, center, _track_history, stop_line_px)
            if check_red_light_violation(sl_result, signal_state):
                violations_found.append({
                    "violation_type": "RED_LIGHT_VIOLATION",
                    "vehicle_bbox": veh.bbox,
                    "vehicle_type": veh.label,
                    "confidence": 0.8,
                    "signal_state": signal_state,
                    "stub_mode": False,
                })
            elif sl_result.is_violation:
                violations_found.append({
                    "violation_type": "STOP_LINE_VIOLATION",
                    "vehicle_bbox": veh.bbox,
                    "vehicle_type": veh.label,
                    "confidence": 0.65,
                    "stub_mode": False,
                })

            park_result = check_illegal_parking(veh.track_id, center, _track_history, no_parking_px, now_ts)
            if park_result.is_violation:
                violations_found.append({
                    "violation_type": "ILLEGAL_PARKING",
                    "vehicle_bbox": veh.bbox,
                    "vehicle_type": veh.label,
                    "confidence": 0.75,
                    "dwell_seconds": park_result.dwell_seconds,
                    "stub_mode": False,
                })

            ws = check_wrong_side(veh.track_id, _track_history, DEMO_CAMERA_CONFIG.lane_direction)
            if ws.is_violation:
                violations_found.append({
                    "violation_type": "WRONG_SIDE_DRIVING",
                    "vehicle_bbox": veh.bbox,
                    "vehicle_type": veh.label,
                    "confidence": 0.7,
                    "angle_deviation": ws.angle_deviation_deg,
                    "stub_mode": False,
                })

    timing_ms["3_violation_rules"] = round((time.perf_counter() - _t0) * 1000, 2)
    steps["4_violations_detected"] = violations_found

    # Step: plate recognition + evidence generation + DB persistence
    _t0 = time.perf_counter()
    persisted = []
    for v in violations_found:
        plate_result = recognize_plate(processed_frame, v["vehicle_bbox"])

        evidence = generate_evidence(
            frame=processed_frame,
            vehicle_bbox=v["vehicle_bbox"],
            vehicle_type=v["vehicle_type"],
            plate_number=plate_result.plate_number,
            violation_type=v["violation_type"],
            confidence=v["confidence"],
            camera_id=camera_id,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            plate_bbox=plate_result.plate_bbox,
        )

        prior_count = crud.count_prior_violations_for_plate(db, plate_result.plate_number)
        risk = compute_risk(v["violation_type"], prior_count, None)

        vehicle_row = crud.get_or_create_vehicle(db, plate_result.plate_number, v["vehicle_type"])

        violation_row = crud.create_violation(
            db,
            vehicle_id=vehicle_row.id,
            vehicle_type=v["vehicle_type"],
            plate_number=plate_result.plate_number,
            violation_type=v["violation_type"],
            confidence=v["confidence"],
            timestamp=dt.datetime.utcnow(),
            camera_id=camera_id,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            evidence_image_path=evidence["evidence_image_path"],
            evidence_json_path=evidence["evidence_json_path"],
            risk_score=risk.risk_score,
            risk_category=risk.risk_category,
            enforcement_priority=risk.enforcement_priority,
            notes=risk.explanation,
        )

        persisted.append({
            **v,
            "violation_id": violation_row.violation_id,
            "plate_number": plate_result.plate_number,
            "plate_ocr_confidence": plate_result.ocr_confidence,
            "plate_localization_method": plate_result.localization_method,
            "risk_score": risk.risk_score,
            "risk_category": risk.risk_category,
            "enforcement_priority": risk.enforcement_priority,
            "evidence_image_path": evidence["evidence_image_path"],
            "evidence_json_path": evidence["evidence_json_path"],
        })

    timing_ms["4_plate_ocr_and_evidence"] = round((time.perf_counter() - _t0) * 1000, 2)
    steps["5_plate_and_evidence"] = persisted
    steps["6_db_updated"] = True

    timing_ms["total"] = round(sum(timing_ms.values()), 2)

    return {
        "camera_id": camera_id,
        "frame_shape": {"width": w, "height": h},
        "steps": steps,
        "violations": persisted,
        "total_detections": len(detections),
        "total_violations": len(persisted),
        "timing_ms": timing_ms,
    }


def reset_session():
    """Call before processing a new, unrelated video so track IDs and
    dwell timers don't bleed across clips during back-to-back demos."""
    global _track_history
    _detector.reset_tracker()
    _track_history = TrackHistory()
