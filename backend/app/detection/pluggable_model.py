"""
GridlockVision AI — Pluggable model loader.

This is the single most important honesty mechanism in the codebase:
helmet, seatbelt, and plate-detection models need fine-tuning on labelled
data we don't have at hackathon-build time. Rather than fabricate a
"working" detector with fake confidence numbers, every such module:

  1. Tries to load a real .pt weight file from the configured path.
  2. If found -> runs real inference, real confidence scores.
  3. If NOT found -> returns an explicit `UNKNOWN` status with
     confidence=0.0 and `stub_mode=True`, and the API/dashboard render
     this distinctly (gray "not yet trained" badge, not a fake green tick).

This means the *pipeline* (detect -> crop -> classify -> log -> evidence)
is 100% real and demoable end-to-end on day one. Only the classification
*accuracy* depends on training data you add later. This separation is
exactly what's needed for a hackathon: the architecture is complete,
the model weights are a clearly marked plug point.
"""
import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


class PluggableYOLO:
    """Wraps a YOLO model that may or may not have trained weights yet."""

    def __init__(self, weight_path: str, task_name: str):
        self.task_name = task_name
        self.weight_path = weight_path
        self.is_trained = Path(weight_path).exists()
        self.model: Optional[YOLO] = None

        if self.is_trained:
            self.model = YOLO(weight_path)
        else:
            # Not an error — this is the expected state before the team
            # trains/drops in a fine-tuned model. Logged once, not spammed.
            print(
                f"[GridlockVision] NOTICE: '{task_name}' model not found at "
                f"{weight_path}. Running in STUB MODE — this module will "
                f"report UNKNOWN instead of fabricating detections. "
                f"Train a model and place it at this path to activate."
            )

    def predict(self, crop, conf_threshold: float = 0.4):
        """Returns list of (label, confidence, bbox) if trained, else []
        with stub_mode flagged by the caller (checked via self.is_trained).
        """
        if not self.is_trained or self.model is None:
            return []
        results = self.model.predict(crop, conf=conf_threshold, verbose=False)
        out = []
        if results and results[0].boxes is not None:
            r = results[0]
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                label = self.model.names.get(cls_id, "unknown")
                conf = float(r.boxes.conf[i])
                bbox = tuple(float(v) for v in r.boxes.xyxy[i])
                out.append((label, conf, bbox))
        return out
