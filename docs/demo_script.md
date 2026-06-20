# Demo Script — GridlockVision AI (Target: 5-7 minutes)

## 0. Setup (before judges arrive)
- `docker compose up --build` running, both services healthy.
- Have 2-3 sample images/clips ready on a USB or laptop folder:
  one daytime clear shot, one low-light/night shot, one with multiple
  motorcycles for triple-riding.
- Open dashboard in browser, on **Judge Demo Mode** tab.

## 1. Opening (30 sec)
"GridlockVision AI automates traffic violation detection from camera
footage — built end-to-end: detection, violation logic, plate OCR,
evidence generation, and an analytics dashboard, all running live, no
internet dependency after first load."

## 2. Live Demo — Image (90 sec)
1. Upload the night/low-light sample image.
2. Point at **Step 1: Preprocessing** — call out the brightness/blur
   scores and which corrections fired automatically.
3. Point at **Step 2: Detection** — bounding boxes + track IDs +
   confidence scores rendered.
4. Point at **Step 3: Signal State** if a signal is in frame.
5. Point at **Step 4-6** — any violations found, the plate OCR result,
   risk score, and the generated evidence card (annotated image +
   metadata).

**If helmet/seatbelt show "STUB MODE"**: say this directly — "this
module is architecturally complete and will activate the moment we
plug in a model trained on our labelled dataset; we chose not to fake a
result here." This is a *strength* to state, not a weakness to hide.

## 3. Live Demo — Triple Riding / Parking (60 sec)
Upload the multi-rider image or a parking-lot shot. Show the
rider-count / dwell-time numbers in the violation card explaining why
it fired.

## 4. Analytics Dashboard (90 sec)
Switch tabs. Walk through:
- Totals + by-type bar chart
- Daily/hourly trend lines
- Heatmap (GPS-tagged violations)
- AI Traffic Insights — read 1-2 generated sentences aloud
- Predicted hotspots table
- Search by plate number live

## 5. Innovation Callouts (45 sec)
- Risk Score formula is transparent and auditable (open `risk_engine.py`
  if asked) — not a black box.
- Insights are real statistics over actual stored data, not templated
  fiction.
- Everything is Dockerized; `docker compose up` is the entire deploy.

## 6. Close (30 sec)
"Every component you saw is real, working code — not slides. The exact
boundary between what's trained today and what needs labelled data is
documented in our README, because a hackathon judge should be able to
trust every claim we make."

## Fallback if live demo fails
Have 2-3 pre-generated evidence PNGs + a terminal recording (asciinema
or screen capture) as backup. State clearly if switching to backup:
"environment hiccup, here's a recording of the same flow."
