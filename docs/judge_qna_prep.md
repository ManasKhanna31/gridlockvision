# Judge Q&A Preparation

Anticipated questions, with honest, confident answers. Bracketed notes
are speaking tips, not part of the answer itself.

---

**Q: Is this actually working, or is it a mockup?**
A: It's working end-to-end. Preprocessing, detection, tracking, all seven
violation rules' logic, plate OCR pipeline, evidence generation, database,
and dashboard all run on real code with no fabricated outputs. The one
honest caveat: helmet, seatbelt, and trained-plate-detector accuracy
depends on fine-tuning weights we haven't trained yet (no labelled Indian
traffic dataset in hackathon timeframe) — those modules report `UNKNOWN`
instead of guessing, which you can see live in the demo.

**Q: Why are helmet/seatbelt detection not "real" yet?**
A: They need a model fine-tuned on labelled images of helmeted/unhelmeted
riders and belted/unbelted drivers. We don't have that dataset built yet.
Rather than fake a result with a hardcoded "looks helmeted" guess, the
module is architected as a pluggable model slot — drop in a `.pt` file
trained on, say, a Roboflow helmet-detection dataset, and it activates
with zero other code changes. We'd rather show you an honest gap than a
dishonest demo.

**Q: How does the Risk Score work — is it ML?**
A: No, deliberately not. It's a transparent, auditable formula: severity
weight by violation type + frequency of prior violations + recency. We
chose this over a black-box model because (a) we don't have enough
labelled outcome data to train a real risk model yet, and (b) a formula
a judge or police officer can read and challenge is more trustworthy for
an enforcement tool than an opaque score.

**Q: How accurate is the system? What's your mAP/F1?**
A: We won't quote a number we can't back with a labelled test set — that
would be a fabricated metric, which the brief explicitly asked us to
avoid. We built a real evaluation harness (`tests/evaluate.py`) that
computes Precision/Recall/F1/mAP the moment you point it at a labelled
test set; for the vehicle-detection layer (which uses pretrained YOLOv8
COCO weights, not a custom model), that's something we can compute live
on a small annotated sample if useful.

**Q: Does this work on Indian-specific vehicles like auto-rickshaws?**
A: Partially, and we're upfront about the limitation. The base detector
uses COCO classes, which don't include auto-rickshaw. We approximate it
via an aspect-ratio heuristic on motorcycle-class detections, and we flag
every such detection with `vehicle_type_is_approximated: true` so it's
never silently treated as ground truth. Fine-tuning on the India Driving
Dataset (IDD) is the documented next step to get a real auto-rickshaw
class.

**Q: How does red-light violation detection work technically?**
A: Two combined signals: (1) we read the traffic signal's color via HSV
thresholding on a configured ROI box — no model needed, works well in
decent lighting; (2) we track each vehicle's position frame-to-frame and
detect when its center crosses the stop-line segment. If the crossing
happens while the signal reads red, that's flagged. We use a line-segment
sign-flip test for crossing, which is geometrically exact, not a
heuristic guess.

**Q: Can this scale to real-time multi-camera deployment?**
A: The current build processes uploaded images/video, which is the right
scope for a judged demo. Architecturally it's modular: the tracker state
is per-camera, ROI config is per-camera in the DB schema already, and the
FastAPI backend can be horizontally scaled behind a queue (e.g. add a
Celery/Redis layer ahead of the pipeline) for live RTSP ingestion — that's
explicitly listed as a future enhancement, not something we're claiming
works today.

**Q: What happens with bad weather / low light?**
A: That's specifically why we built an always-on preprocessing stage:
CLAHE-based low-light enhancement, unsharp-mask deblurring, shadow removal
via LAB-space gamma correction, and rain-streak suppression via median +
bilateral filtering — all classical CV, so it works without any extra
model and without internet. You can see the before/after brightness and
blur scores live in Demo Mode Step 1.

**Q: Why SQLite and not a "real" database?**
A: SQLite is correct for a hackathon demo — zero setup, file-based, fast
to show. The schema (SQLAlchemy ORM) is database-agnostic; switching to
PostgreSQL is a one-line change to `DATABASE_URL` in `core/config.py`,
no other code changes. We chose to spend our limited build time on
detection logic rather than infra ceremony that wouldn't change the demo.

**Q: What's your biggest technical risk if you had to keep building this?**
A: Getting a labelled Indian traffic dataset for helmet/seatbelt/plate
fine-tuning. Everything else in the pipeline is solid engineering; that
data-collection-and-annotation step is the real bottleneck to a
production-accurate system, and we'd rather say that clearly than pretend
otherwise.

---

## Quick technical facts to have ready
- Detection: YOLOv8n (nano) pretrained on COCO, tracked via Ultralytics'
  built-in ByteTrack integration.
- OCR: EasyOCR, English, CPU-mode by default (GPU optional).
- DB: SQLite by default, PostgreSQL-ready.
- Stack: FastAPI + Streamlit + OpenCV + SQLAlchemy + Docker Compose.
- Lines of original Python code: ~2,350 across backend + dashboard
  (verified via `wc -l` on the repo).
