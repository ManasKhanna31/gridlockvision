# 🚦 GridlockVision AI

**Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**
Flipkart Gridlock Hackathon 2.0 — Theme 3 Submission

---

## 1. What This Is

A working, demoable prototype that detects road users in traffic camera
footage, flags seven categories of traffic violations using a mix of
trained CV models and explainable rule-based logic, OCRs the license
plate, generates timestamped evidence, scores violation risk, and
surfaces everything on an analytics dashboard — built to be run live in
front of judges via **Judge Demo Mode**.

It is built honestly: every module that needs a fine-tuned model
(helmet, seatbelt, plate detection) runs in a clearly-labeled **stub
mode** until you train and drop in real weights. Nothing fabricates a
detection or a metric. See [Section 6](#6-implemented-vs-future-enhancements)
for the exact line between what works today and what needs your training
data.

---

## 2. Quick Start

### Option A — Docker (recommended for judges)

```bash
cd docker
docker compose up --build
```

- Backend API: http://localhost:8000 (docs at `/docs`)
- Dashboard: http://localhost:8501

### Option B — Local Python

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — API
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Dashboard
cd frontend/dashboard
streamlit run app.py
```

First run downloads the pretrained YOLOv8n weights (~6MB) automatically
via `ultralytics` — needs internet once, then it's cached locally.

### Try it immediately

1. Open the dashboard → **Judge Demo Mode**.
2. Upload any traffic photo (a street scene with cars/bikes works fine
   for the vehicle-detection + triple-riding + parking/stop-line logic;
   helmet/seatbelt/plate accuracy depends on whether you've trained
   those models — see Section 6).
3. Watch Steps 1–6 render live, then switch to **Analytics Dashboard**
   to see it land in the database, charts, and heatmap.

---

## 3. Architecture

```
                    ┌─────────────────────┐
                    │   Streamlit          │
                    │   Dashboard          │  <- judges interact here
                    │ (Demo Mode +         │
                    │  Analytics)          │
                    └──────────┬───────────┘
                               │ HTTP
                    ┌──────────▼───────────┐
                    │   FastAPI Backend     │
                    │  /demo  /violations   │
                    │  /analytics           │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        │                      │                        │
┌───────▼────────┐   ┌─────────▼─────────┐   ┌──────────▼─────────┐
│ Preprocessing   │   │ Detection +        │   │ Violation Rules     │
│ (CLAHE, deblur, │──▶│ Tracking           │──▶│ (helmet, seatbelt,  │
│ shadow/rain,    │   │ (YOLOv8 + ByteTrack)│   │ triple-ride, wrong- │
│ normalize)      │   │                    │   │ side, stop-line,    │
└─────────────────┘   └────────────────────┘   │ red-light, parking) │
                                                 └──────────┬──────────┘
                                                            │
                                       ┌────────────────────▼───────────────────┐
                                       │ Plate OCR → Evidence Gen → Risk Score   │
                                       │ → SQLite/PostgreSQL                     │
                                       └─────────────────┬───────────────────────┘
                                                          │
                                                ┌─────────▼─────────┐
                                                │ Analytics Engine    │
                                                │ (trends, heatmap,   │
                                                │ AI insights,        │
                                                │ hotspot prediction) │
                                                └────────────────────┘
```

A standalone diagram (`docs/architecture_diagram.md`) and workflow
sequence diagram (`docs/workflow_diagram.md`) are included as Mermaid
source for slide decks.

---

## 4. Project Structure

```
gridlockvision/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI entrypoint
│       ├── core/
│       │   ├── config.py            # all paths, thresholds, ROI defaults
│       │   └── pipeline.py          # orchestrator: frame -> violations
│       ├── api/routes/
│       │   ├── demo.py              # Judge Demo Mode endpoints
│       │   ├── violations.py        # search/retrieve violations
│       │   └── analytics.py         # dashboard data endpoints
│       ├── preprocessing/
│       │   └── pipeline.py          # low-light, deblur, contrast, rain/shadow
│       ├── detection/
│       │   ├── vehicle_detector.py  # YOLOv8 + ByteTrack road-user detection
│       │   └── pluggable_model.py   # honest stub/trained model loader
│       ├── violations/
│       │   ├── helmet_check.py
│       │   ├── seatbelt_check.py
│       │   ├── triple_riding.py
│       │   ├── trajectory_rules.py  # wrong-side, stop-line, red-light, parking
│       │   ├── roi_config.py        # stop-line/no-parking/lane geometry
│       │   ├── risk_engine.py       # Innovation A & B: risk score + priority
│       │   ├── insights_engine.py   # Innovation C & D: AI insights + hotspots
│       │   └── evidence_generator.py
│       ├── ocr/
│       │   └── plate_recognition.py # plate localization + deskew + EasyOCR
│       └── db/
│           ├── models.py            # SQLAlchemy schema
│           └── crud.py
├── frontend/dashboard/
│   └── app.py                       # Streamlit Demo Mode + Analytics UI
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── tests/
│   └── evaluate.py                  # real-metrics evaluation harness
├── docs/
│   ├── architecture_diagram.md
│   ├── workflow_diagram.md
│   ├── demo_script.md
│   ├── judge_qna_prep.md
│   └── api_reference.md
├── data/                             # uploads, evidence, sample videos, db
└── requirements.txt
```

---

## 5. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/demo/image` | Run full pipeline on one uploaded image |
| POST | `/demo/video` | Run full pipeline on uploaded video (frame-sampled) |
| GET | `/violations/search` | Filter by plate, type, date range |
| GET | `/violations/{violation_id}` | Single violation detail |
| GET | `/analytics/summary` | Totals, by-type, by-camera, top offenders |
| GET | `/analytics/trends/daily` | Daily violation counts |
| GET | `/analytics/trends/hourly` | Hourly violation counts |
| GET | `/analytics/heatmap` | GPS-tagged points for map visualization |
| GET | `/analytics/insights` | AI-generated natural-language trend summaries |
| GET | `/analytics/hotspots` | Predicted high-risk camera locations |

Full interactive docs at `/docs` once the backend is running (FastAPI
auto-generates this from the code — nothing to maintain separately).

---

## 6. Implemented vs. Future Enhancements

This separation is the most important section for judges — read it
before asking "does X really work?"

### ✅ Fully implemented and runnable today (no training data needed)
- Image preprocessing: low-light (CLAHE), deblur (unsharp mask), contrast,
  shadow removal, rain-noise reduction, normalization — all classical CV,
  always on, real algorithms.
- Road-user detection + multi-object tracking (YOLOv8 pretrained on COCO
  + ByteTrack) for car/motorcycle/bus/truck/bicycle/pedestrian.
- Triple riding (geometric rider-count heuristic).
- Wrong-side driving (trajectory-vector vs. lane-direction angle check).
- Stop-line violation (line-crossing detection via track history).
- Red-light violation (HSV-based signal color read + stop-line crossing).
- Illegal parking (dwell-time tracking inside a configurable no-parking
  ROI).
- License plate localization (classical CV fallback) + skew correction +
  EasyOCR text extraction + Indian-format validation.
- Evidence generation: annotated PNG + JSON metadata, exactly matching
  the brief's schema.
- Full violation database (SQLite, swappable to PostgreSQL via one
  config line).
- Analytics dashboard: totals, by-type/by-camera breakdowns, daily/hourly
  trends, heatmap, top-offender ranking, plate/date search.
- Judge Demo Mode with step-by-step visual pipeline trace.
- Risk Score + Enforcement Recommendation (transparent, auditable
  formula — see `risk_engine.py`).
- AI Traffic Insights (real statistical trend detection over your own
  violation data, not templated fiction).
- Hotspot ranking (frequency-based heuristic over historical data).
- Docker deployment for both services.
- Evaluation harness that computes **real** Precision/Recall/F1/mAP —
  but only once you supply a labeled test set (see `tests/evaluate.py`).

### 🔶 Architecturally complete, accuracy depends on training data you add
These run end-to-end today and return an honest `UNKNOWN` / `stub_mode:
true` instead of a fabricated result until trained weights are placed in
`backend/app/models_store/`:
- **Helmet detection** — needs a helmet/no-helmet classifier fine-tuned
  on cropped rider-head images (e.g. labelled subset of a public Indian
  traffic dataset, or self-collected + Roboflow/LabelImg annotation).
- **Seatbelt detection** — same pattern, needs a seatbelt/no-seatbelt
  classifier on cropped driver regions.
- **Trained plate detector** — current fallback is a classical
  CV contour locator (works, but lower recall on tilted/dirty/occluded
  plates than a trained model would give).

**To activate:** train a YOLOv8 classification/detection model on your
labelled data, export the `.pt` file, and place it at the path printed
in the startup console log (or set the corresponding env var in
`core/config.py`). No other code changes needed — the rest of the
pipeline already calls these modules correctly.

### 🔷 Honest future enhancements (not built, won't claim otherwise)
- Learned deblurring (DeblurGAN-style) for severe motion blur.
- Learned de-raining network for heavy rain streaks.
- Auto-rickshaw as a real trained class (currently an aspect-ratio
  heuristic on motorcycle detections — flagged via
  `vehicle_type_is_approximated`).
- Pose-estimation-based rider counting for heavily occluded triple-riding
  cases.
- Trained signal-color classifier to replace HSV thresholding for
  glare/dusk robustness.
- True time-series hotspot forecasting (Prophet/ARIMA) once enough
  historical data exists to fit one meaningfully.
- Live RTSP camera ingestion (current scope is upload-based, which is
  the right scope for a judged demo).

---

## 7. Installation Guide

See [Section 2](#2-quick-start). Requirements: Python 3.10+, ~2GB free
disk for model caches, Docker (optional but recommended).

## 8. Evaluation (Real Metrics)

```bash
cd tests
python evaluate.py --mode detection --images data/test/images --labels data/test/labels
python evaluate.py --mode violations --csv data/test/violation_ground_truth.csv
```

The script **refuses to print a number without ground truth** — this is
intentional. See the script's docstring for how to build a small labeled
test set in under an hour.

## 9. Tech Stack

Python · FastAPI · YOLOv8 (Ultralytics, with ByteTrack) · OpenCV ·
EasyOCR · Streamlit · SQLite (PostgreSQL-ready) · Docker

## 10. Team Notes for Judges

This was built to be realistically completable by a B.Tech AI/ML student
hackathon team: every "implemented" item above is real, runnable code in
this repo, not a slide. The honest stub-mode pattern for helmet/seatbelt/
plate models is a deliberate engineering choice, not a shortcut — it
means the full system architecture, data flow, database, dashboard, and
demo experience are complete and judgeable today, while the genuinely
hard part (collecting and labeling Indian traffic data) is correctly
scoped as the next sprint, not glossed over.
