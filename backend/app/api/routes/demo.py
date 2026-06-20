"""
GridlockVision AI — Demo Mode API routes.
Implements the brief's Judge Demo Mode: upload image/video -> full pipeline.
"""
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, UploadFile, File, Depends, Form
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_DIR, DEFAULT_CAMERA_ID, DEFAULT_GPS_LAT, DEFAULT_GPS_LON
from app.core.pipeline import process_frame, reset_session
from app.db.models import get_db

router = APIRouter(prefix="/demo", tags=["Demo Mode"])


@router.post("/image")
async def demo_process_image(
    file: UploadFile = File(...),
    camera_id: str = Form(DEFAULT_CAMERA_ID),
    gps_lat: float = Form(DEFAULT_GPS_LAT),
    gps_lon: float = Form(DEFAULT_GPS_LON),
    db: Session = Depends(get_db),
):
    """Step-by-step single-image demo: returns every pipeline stage so the
    frontend can render "Step 1 -> Step 6" visually, per the brief.
    """
    ext = Path(file.filename).suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    frame = cv2.imread(str(save_path))
    if frame is None:
        return {"error": "Could not read uploaded image. Supported: jpg, png, bmp."}

    reset_session()
    result = process_frame(frame, db, camera_id=camera_id, gps_lat=gps_lat, gps_lon=gps_lon, is_video_frame=False)
    return result


@router.post("/video")
async def demo_process_video(
    file: UploadFile = File(...),
    camera_id: str = Form(DEFAULT_CAMERA_ID),
    gps_lat: float = Form(DEFAULT_GPS_LAT),
    gps_lon: float = Form(DEFAULT_GPS_LON),
    sample_every_n_frames: int = Form(5),
    max_frames: int = Form(150),
    db: Session = Depends(get_db),
):
    """Processes an uploaded video, sampling every Nth frame (full-frame
    processing on every frame of real video is unnecessary for a demo and
    would slow live judging) and tracking objects across the sampled
    sequence for trajectory-based rules.

    max_frames caps total processing time for the live-demo time budget.
    """
    ext = Path(file.filename).suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(save_path))
    if not cap.isOpened():
        return {"error": "Could not open uploaded video."}

    reset_session()
    frame_idx = 0
    processed_count = 0
    all_violations = []
    last_result = None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n_frames == 0:
            ts = frame_idx / fps
            result = process_frame(
                frame, db, camera_id=camera_id, gps_lat=gps_lat, gps_lon=gps_lon,
                is_video_frame=True, frame_ts=ts,
            )
            all_violations.extend(result["violations"])
            last_result = result
            processed_count += 1
            if processed_count >= max_frames:
                break
        frame_idx += 1

    cap.release()

    return {
        "camera_id": camera_id,
        "total_frames_sampled": processed_count,
        "total_violations": len(all_violations),
        "violations": all_violations,
        "last_frame_steps": last_result["steps"] if last_result else None,
    }
