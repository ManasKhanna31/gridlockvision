"""
GridlockVision AI — Evaluation harness.

IMPORTANT, READ BEFORE QUOTING ANY METRIC TO JUDGES:
This script computes REAL metrics, but only if you provide a labeled
ground-truth test set. There is no shortcut around this — any number
printed without ground-truth data behind it would be fabricated, which
the project brief explicitly says to avoid, and we don't.

Two evaluation modes:

1. Detection metrics (Precision, Recall, F1, mAP@0.5) for the vehicle
   detector — needs a YOLO-format labeled test set (images + .txt label
   files with class_id x_center y_center width height, normalized).
   Point TEST_IMAGES_DIR / TEST_LABELS_DIR at your annotated folder
   (e.g. a small slice of IDD or a manually labeled 50-100 image sample)
   and run:
       python evaluate.py --mode detection --images <dir> --labels <dir>

2. Violation-rule accuracy — needs a CSV of (image_path, true_violation_
   labels) you annotate by hand for a held-out demo clip. Run:
       python evaluate.py --mode violations --csv ground_truth.csv

WHAT THIS SCRIPT DOES NOT DO:
It does not invent numbers when no ground truth is supplied — it exits
with an explanation instead. This is intentional per the brief's
"Avoid fake metrics" requirement.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.detection.vehicle_detector import VehicleDetector
import cv2


def _iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_detection(images_dir: Path, labels_dir: Path, iou_threshold: float = 0.5):
    if not images_dir.exists() or not labels_dir.exists():
        print(
            "ERROR: labeled test set not found.\n"
            f"  images_dir={images_dir} exists={images_dir.exists()}\n"
            f"  labels_dir={labels_dir} exists={labels_dir.exists()}\n\n"
            "No metrics can be computed without ground-truth labels. "
            "This script will NOT print fabricated numbers. Annotate a "
            "small test set (even 30-50 images via a tool like LabelImg "
            "or Roboflow, YOLO .txt format) and re-run."
        )
        sys.exit(1)

    detector = VehicleDetector()
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if not image_paths:
        print(f"ERROR: no images found in {images_dir}")
        sys.exit(1)

    tp, fp, fn = 0, 0, 0
    ious_for_map = []

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    gt_boxes.append((x1, y1, x2, y2))

        detections = detector.detect(frame)
        pred_boxes = [d.bbox for d in detections]

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
                ious_for_map.append(best_iou)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    approx_map50 = np.mean(ious_for_map) * precision if ious_for_map else 0.0

    print("=" * 50)
    print("DETECTION EVALUATION (real, computed from your test set)")
    print("=" * 50)
    print(f"Images evaluated : {len(image_paths)}")
    print(f"True Positives   : {tp}")
    print(f"False Positives  : {fp}")
    print(f"False Negatives  : {fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Approx. mAP@0.5  : {approx_map50:.4f}  "
          f"(simplified single-IoU-threshold approximation — for a rigorous "
          f"mAP, use Ultralytics' built-in `model.val()` on a YOLO-format "
          f"dataset.yaml, which computes proper precision-recall curves "
          f"across confidence thresholds.)")


def evaluate_violations(csv_path: Path):
    """Expects a CSV with columns: image_path, true_label (0/1), pred_label
    pre-computed by running Demo Mode on each image and recording whether
    the system flagged the violation a human annotator confirmed.
    """
    if not csv_path.exists():
        print(
            f"ERROR: {csv_path} not found. No metrics computed.\n"
            "Create a ground-truth CSV (image_path,true_label,pred_label) "
            "by manually reviewing a held-out demo clip's outputs against "
            "what a human would judge as a real violation."
        )
        sys.exit(1)

    y_true, y_pred = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["true_label"]))
            y_pred.append(int(row["pred_label"]))

    if not y_true:
        print("ERROR: CSV had no rows.")
        sys.exit(1)

    print("=" * 50)
    print("VIOLATION-RULE EVALUATION (real, computed from your CSV)")
    print("=" * 50)
    print(f"Samples          : {len(y_true)}")
    print(f"Accuracy         : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision        : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall           : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score         : {f1_score(y_true, y_pred, zero_division=0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridlockVision AI evaluation harness")
    parser.add_argument("--mode", choices=["detection", "violations"], required=True)
    parser.add_argument("--images", type=str, default="data/test/images")
    parser.add_argument("--labels", type=str, default="data/test/labels")
    parser.add_argument("--csv", type=str, default="data/test/violation_ground_truth.csv")
    args = parser.parse_args()

    if args.mode == "detection":
        evaluate_detection(Path(args.images), Path(args.labels))
    else:
        evaluate_violations(Path(args.csv))
