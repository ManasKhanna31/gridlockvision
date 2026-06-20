# API Reference

Base URL (local): `http://localhost:8000`
Interactive Swagger docs: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

---

## Demo Mode

### `POST /demo/image`
Runs the full pipeline on one image.

**Form fields:**
| Field | Type | Default |
|---|---|---|
| `file` | file (jpg/png/bmp) | required |
| `camera_id` | string | `CAM-DEMO-01` |
| `gps_lat` | float | `28.6139` |
| `gps_lon` | float | `77.2090` |

**Response (200):**
```json
{
  "camera_id": "CAM-DEMO-01",
  "frame_shape": {"width": 1280, "height": 720},
  "steps": {
    "1_preprocessing": {"low_light_enhancement": true, "...": "..."},
    "2_detection": [{"label": "car", "confidence": 0.91, "bbox": [...], "track_id": 1}],
    "3_signal_state": "RED",
    "4_violations_detected": [...],
    "5_plate_and_evidence": [...],
    "6_db_updated": true
  },
  "violations": [...],
  "total_detections": 4,
  "total_violations": 1
}
```

### `POST /demo/video`
Same pipeline run across sampled frames of an uploaded video.

**Additional form fields:** `sample_every_n_frames` (default 5),
`max_frames` (default 150).

---

## Violations

### `GET /violations/search`
Query params: `plate_number`, `violation_type`, `date_from` (ISO 8601),
`date_to` (ISO 8601), `limit` (default 200, max 1000).

### `GET /violations/{violation_id}`
Full detail for a single violation record.

---

## Analytics

| Endpoint | Returns |
|---|---|
| `GET /analytics/summary` | totals, by-type, by-camera, top offenders |
| `GET /analytics/trends/daily?days=14` | daily counts |
| `GET /analytics/trends/hourly` | hourly counts (0-23) |
| `GET /analytics/heatmap` | GPS-tagged violation points |
| `GET /analytics/insights?lookback_days=7` | AI-generated trend sentences |
| `GET /analytics/hotspots?top_n=5` | predicted high-risk camera locations |

---

## Static Evidence Files
Generated evidence images are served at:
`GET /evidence/{violation_id}.png`
