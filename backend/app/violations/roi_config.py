"""
GridlockVision AI — ROI / lane configuration.

Defines the spatial reference geometry that violation rules need:
  - stop line (a line segment)
  - no-parking zone (a polygon)
  - lane direction vector (expected direction of travel, for wrong-side
    detection)
  - traffic-signal ROI (a small box where the camera can read signal color)

In production each camera has its own calibrated config (stored in the
`cameras` DB table, see db/models.py). For the hackathon demo, one
preconfigured profile is provided here as DEMO_CAMERA_CONFIG, expressed
in *normalized* coordinates (0.0-1.0 of frame width/height) so it scales
to any input resolution — this is important since judges may upload
arbitrary video/image sizes during the live demo.

The Streamlit dashboard exposes a simple ROI-drawing tool (see
frontend/dashboard/roi_editor.py) so the configuration is not hardcoded
black-box logic — judges can see and even redraw it live.
"""
from dataclasses import dataclass, field


@dataclass
class CameraROIConfig:
    camera_id: str
    # stop line: two points, normalized (x, y) 0-1
    stop_line: tuple = ((0.1, 0.55), (0.9, 0.55))
    # no-parking polygon: list of normalized (x, y) points
    no_parking_zone: list = field(default_factory=lambda: [
        (0.65, 0.6), (0.95, 0.6), (0.95, 0.95), (0.65, 0.95)
    ])
    # expected lane direction vector, normalized, pointing "downstream"
    lane_direction: tuple = (0.0, 1.0)  # top-to-bottom traffic flow
    # traffic signal ROI box: (x1, y1, x2, y2) normalized
    signal_roi: tuple = (0.42, 0.02, 0.58, 0.18)


DEMO_CAMERA_CONFIG = CameraROIConfig(camera_id="CAM-DEMO-01")


def denormalize_point(point, frame_w: int, frame_h: int):
    x, y = point
    return (x * frame_w, y * frame_h)


def denormalize_line(line, frame_w: int, frame_h: int):
    return (
        denormalize_point(line[0], frame_w, frame_h),
        denormalize_point(line[1], frame_w, frame_h),
    )


def denormalize_polygon(polygon, frame_w: int, frame_h: int):
    return [denormalize_point(p, frame_w, frame_h) for p in polygon]


def denormalize_box(box, frame_w: int, frame_h: int):
    x1, y1, x2, y2 = box
    return (x1 * frame_w, y1 * frame_h, x2 * frame_w, y2 * frame_h)
