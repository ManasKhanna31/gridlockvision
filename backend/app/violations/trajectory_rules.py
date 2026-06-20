"""
GridlockVision AI — Trajectory & ROI-based violation rules.

All four checks below are IMPLEMENTED with real, working logic operating
on tracked detections across frames (track_id from VehicleDetector). They
need a short history of positions per track_id, which TrackHistory below
maintains.

  D. Wrong-side driving  -> compare each track's movement vector against
                            the configured lane direction; flag if the
                            angle deviation exceeds a tolerance.
  E. Stop-line violation -> flag when a track's bbox center crosses the
                            stop-line segment while... (see red light
                            note: stop-line crossing alone is *parking/
                            stopping* enforcement; combined with red-light
                            state it becomes a red-light violation).
  F. Red-light violation -> reads the signal-state ROI color (red/yellow/
                            green via HSV thresholding — classical CV,
                            no model needed) and flags vehicles crossing
                            the stop line while the signal reads red.
  G. Illegal parking      -> tracks dwell time: how many consecutive
                            seconds a track's center has stayed inside
                            the no-parking polygon below a movement
                            threshold.

FUTURE ENHANCEMENT: signal-color reading via HSV is lighting-sensitive;
a small trained classifier on cropped signal-head images would be more
robust under glare/dusk and is a natural v2 upgrade.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np

from app.core.config import (
    WRONG_SIDE_ANGLE_TOLERANCE_DEG,
    ILLEGAL_PARK_DWELL_SECONDS,
    STOP_LINE_VIOLATION_BUFFER_PX,
)
from app.violations.roi_config import (
    denormalize_line, denormalize_polygon, denormalize_box
)


# ---------------------------------------------------------------------------
# Track history — shared positional memory across frames
# ---------------------------------------------------------------------------
class TrackHistory:
    def __init__(self, max_len: int = 30):
        self.positions = defaultdict(lambda: deque(maxlen=max_len))
        self.first_seen_in_zone = {}  # track_id -> timestamp when entered no-parking zone
        self.crossed_stop_line = set()  # track_ids already flagged this session

    def update(self, track_id: int, center: tuple, ts: float):
        self.positions[track_id].append((ts, center))

    def get_movement_vector(self, track_id: int):
        hist = self.positions.get(track_id)
        if not hist or len(hist) < 2:
            return None
        (_, p_start) = hist[0]
        (_, p_end) = hist[-1]
        return (p_end[0] - p_start[0], p_end[1] - p_start[1])

    def displacement_magnitude(self, track_id: int) -> float:
        v = self.get_movement_vector(track_id)
        if v is None:
            return 0.0
        return math.hypot(*v)


# ---------------------------------------------------------------------------
# D. Wrong-side driving
# ---------------------------------------------------------------------------
@dataclass
class WrongSideResult:
    is_violation: bool
    angle_deviation_deg: float = 0.0


def check_wrong_side(track_id: int, history: TrackHistory, lane_direction: tuple) -> WrongSideResult:
    vector = history.get_movement_vector(track_id)
    if vector is None or (vector[0] == 0 and vector[1] == 0):
        return WrongSideResult(is_violation=False)

    def angle_of(v):
        return math.degrees(math.atan2(v[1], v[0]))

    vehicle_angle = angle_of(vector)
    lane_angle = angle_of(lane_direction)
    deviation = abs((vehicle_angle - lane_angle + 180) % 360 - 180)

    is_violation = deviation > (180 - WRONG_SIDE_ANGLE_TOLERANCE_DEG)
    return WrongSideResult(is_violation=is_violation, angle_deviation_deg=round(deviation, 2))


# ---------------------------------------------------------------------------
# E. Stop-line violation
# ---------------------------------------------------------------------------
def _point_side_of_line(point, line_p1, line_p2) -> float:
    """Returns signed value: sign indicates which side of the line the
    point is on. Used to detect a sign-flip = crossing event."""
    (x, y), (x1, y1), (x2, y2) = point, line_p1, line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


@dataclass
class StopLineResult:
    is_violation: bool
    crossed_this_frame: bool = False


def check_stop_line(
    track_id: int, center: tuple, history: TrackHistory, stop_line_px: tuple
) -> StopLineResult:
    hist = history.positions.get(track_id)
    if not hist or len(hist) < 2:
        return StopLineResult(is_violation=False)

    prev_center = hist[-2][1]
    side_prev = _point_side_of_line(prev_center, *stop_line_px)
    side_now = _point_side_of_line(center, *stop_line_px)

    crossed = (side_prev * side_now) < 0 and abs(side_now) > STOP_LINE_VIOLATION_BUFFER_PX
    if crossed and track_id not in history.crossed_stop_line:
        history.crossed_stop_line.add(track_id)
        return StopLineResult(is_violation=True, crossed_this_frame=True)
    return StopLineResult(is_violation=False, crossed_this_frame=False)


# ---------------------------------------------------------------------------
# F. Red-light violation (depends on stop-line crossing + signal color)
# ---------------------------------------------------------------------------
def read_signal_state(frame: np.ndarray, signal_box_px: tuple) -> str:
    """Classical HSV color-threshold read of the signal head crop.
    Returns "RED", "YELLOW", "GREEN", or "UNKNOWN".
    """
    x1, y1, x2, y2 = [int(v) for v in signal_box_px]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

    counts = {
        "RED": int(np.count_nonzero(red_mask)),
        "YELLOW": int(np.count_nonzero(yellow_mask)),
        "GREEN": int(np.count_nonzero(green_mask)),
    }
    best = max(counts, key=counts.get)
    if counts[best] < 15:  # too few pixels to be confident
        return "UNKNOWN"
    return best


def check_red_light_violation(stop_line_result: StopLineResult, signal_state: str) -> bool:
    return stop_line_result.crossed_this_frame and signal_state == "RED"


# ---------------------------------------------------------------------------
# G. Illegal parking (dwell time inside no-parking polygon)
# ---------------------------------------------------------------------------
@dataclass
class ParkingResult:
    is_violation: bool
    dwell_seconds: float = 0.0


def _point_in_polygon(point, polygon_px) -> bool:
    contour = np.array(polygon_px, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def check_illegal_parking(
    track_id: int,
    center: tuple,
    history: TrackHistory,
    no_parking_polygon_px: list,
    now_ts: float,
    movement_threshold_px: float = 8.0,
) -> ParkingResult:
    in_zone = _point_in_polygon(center, no_parking_polygon_px)

    if not in_zone:
        history.first_seen_in_zone.pop(track_id, None)
        return ParkingResult(is_violation=False)

    if track_id not in history.first_seen_in_zone:
        history.first_seen_in_zone[track_id] = now_ts
        return ParkingResult(is_violation=False, dwell_seconds=0.0)

    dwell = now_ts - history.first_seen_in_zone[track_id]
    moved = history.displacement_magnitude(track_id)

    # only count dwell as "parked" if the vehicle isn't actively moving through
    if moved > movement_threshold_px * 3:
        history.first_seen_in_zone[track_id] = now_ts  # reset, it's transiting not parking
        return ParkingResult(is_violation=False, dwell_seconds=0.0)

    is_violation = dwell >= ILLEGAL_PARK_DWELL_SECONDS
    return ParkingResult(is_violation=is_violation, dwell_seconds=round(dwell, 1))
