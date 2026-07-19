#!/usr/bin/env python3
"""
Argus Phase 8 — Temperature Scaling Calibration and ECE Evaluation.

Uses COCO-pretrained YOLOv8m on FLIR ADAS validation images to measure
ECE before and after temperature scaling. No additional training required.

Run from project root with venv active:
    python scripts/calibrate_and_evaluate.py

Target: post-calibration ECE < pre-calibration ECE.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.fusion.temperature_scaler import TemperatureScaler, compute_ece

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH    = PROJECT_ROOT / "config.yaml"
MODEL_PATH     = str(PROJECT_ROOT / "argus" / "models" / "yolov8m.pt")
IOU_THRESHOLD  = 0.5


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def iou(box_a, box_b) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def load_flir_annotations(ann_path: Path):
    """Load FLIR ADAS COCO-format annotations. Returns dict {file_name: [boxes]}."""
    with open(ann_path) as fh:
        ann = json.load(fh)
    id_to_file = {img["id"]: img["file_name"] for img in ann.get("images", [])}
    gt = {}
    for a in ann.get("annotations", []):
        if a.get("category_id", 0) != 1:   # person only
            continue
        fname = id_to_file.get(a["image_id"])
        if fname is None:
            continue
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        gt.setdefault(Path(fname).name, []).append(box)
    return gt


def collect_detections_flir(
    detector,
    rgb_dir: Path,
    gt: dict,
    max_images: int,
    confidence_threshold: float,
):
    """Run detector on FLIR ADAS images. Returns (confidences, correct) arrays."""
    image_files = sorted((rgb_dir / "data").glob("*.jpg"))[:max_images]
    confidences, correct = [], []

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        dets = detector.detect(frame)
        gt_boxes = gt.get(img_path.name, [])

        # Detector.detect() returns List[Tuple[x1, y1, x2, y2, conf]] —
        # 5-field tuples confirmed from argus/detection/detector.py:102-154.
        for (x1, y1, x2, y2, conf) in dets:
            if conf < confidence_threshold:
                continue
            pred_box = [x1, y1, x2, y2]
            matched = any(iou(pred_box, gb) >= IOU_THRESHOLD for gb in gt_boxes)
            confidences.append(float(conf))
            correct.append(1 if matched else 0)

    return np.array(confidences, dtype=np.float32), np.array(correct, dtype=int)


def collect_detections_synthetic(n: int = 400):
    """Generate synthetic overconfident detections as fallback."""
    rng = np.random.default_rng(42)
    confidences = np.clip(rng.beta(9, 1, n), 0.5, 0.999).astype(np.float32)
    correct     = (rng.random(n) < 0.60).astype(int)
    return confidences, correct


def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        with mlflow.start_run(
            tags={"pipeline_stage": "fusion_calibration"}
        ):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in params.items():
                mlflow.log_param(k, v)
        logger.info("Metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    config       = load_config()
    cfg          = config["fusion"]
    tracking_uri = config["mlflow"]["tracking_uri"]
    max_images   = cfg["calibration_max_images"]
    n_bins       = cfg["ece_n_bins"]
    det_cfg      = config["detection"]

    # ── Attempt FLIR-based calibration ──
    rgb_val_dir = PROJECT_ROOT / cfg["flir_rgb_val"]
    ann_path    = rgb_val_dir / "index.json"
    source      = "synthetic"

    confidences, correct = None, None

    if ann_path.exists() and (rgb_val_dir / "data").exists():
        try:
            logger.info("Loading FLIR ADAS annotations from %s", ann_path)
            gt = load_flir_annotations(ann_path)

            from argus.detection.detector import Detector
            detector = Detector(
                model_path=MODEL_PATH,
                device=det_cfg["device"],
                confidence_threshold=0.25,
            )
            detector.load_model()

            logger.info(
                "Running YOLOv8m on up to %d FLIR val images...", max_images
            )
            confidences, correct = collect_detections_flir(
                detector, rgb_val_dir, gt, max_images, confidence_threshold=0.25
            )
            source = "flir_adas"
            logger.info(
                "Collected %d detection samples from FLIR ADAS", len(confidences)
            )
        except Exception as exc:
            logger.warning("FLIR collection failed (%s) — using synthetic data.", exc)
            confidences, correct = None, None

    if confidences is None or len(confidences) < 20:
        logger.info("Falling back to synthetic overconfident calibration data.")
        confidences, correct = collect_detections_synthetic(n=400)
        source = "synthetic"

    # ── Shuffle then split: 60% calibration, 40% validation ──
    # Fixed seed ensures reproducibility; shuffle prevents positional bias
    # (detections are collected in image-sorted order, so unsorted split would
    # use first ~120 images for cal and last ~80 for val — different scenes).
    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(confidences))
    confidences = confidences[indices]
    correct     = correct[indices]
    n_cal = int(len(confidences) * 0.6)
    cal_conf,  cal_correct  = confidences[:n_cal],  correct[:n_cal]
    val_conf,  val_correct  = confidences[n_cal:],  correct[n_cal:]

    # ── Pre-calibration ECE ──
    ece_before = compute_ece(val_conf, val_correct.astype(float), n_bins=n_bins)

    # ── Fit temperature scaler ──
    scaler = TemperatureScaler()
    scaler.fit_from_confidences(cal_conf, cal_correct)

    # ── Post-calibration ECE ──
    cal_val_conf = scaler.scale_confidences(val_conf)
    ece_after    = compute_ece(cal_val_conf, val_correct.astype(float), n_bins=n_bins)

    logger.info("─" * 55)
    logger.info("  Source               : %s", source)
    logger.info("  Calibration samples  : %d", n_cal)
    logger.info("  Validation samples   : %d", len(val_conf))
    logger.info("  Temperature T        : %.4f", scaler.temperature)
    logger.info("  ECE before scaling   : %.4f", ece_before)
    logger.info("  ECE after  scaling   : %.4f", ece_after)
    logger.info("  ECE reduction        : %.4f  (%.1f%%)",
                ece_before - ece_after,
                100 * (ece_before - ece_after) / (ece_before + 1e-8))
    logger.info("─" * 55)

    if ece_after < ece_before:
        logger.info("✅  ECE decreased post-calibration: %.4f → %.4f",
                    ece_before, ece_after)
    else:
        logger.warning("⚠️   ECE did NOT decrease: %.4f → %.4f",
                       ece_before, ece_after)

    metrics = {
        "ece_before":   round(ece_before, 4),
        "ece_after":    round(ece_after,  4),
        "ece_reduction": round(ece_before - ece_after, 4),
        "temperature":  round(scaler.temperature, 4),
        "n_samples":    len(confidences),
    }
    params = {
        "source":       source,
        "n_bins":       n_bins,
        "calibration_split": 0.6,
    }
    log_to_mlflow(metrics, params, tracking_uri)


if __name__ == "__main__":
    main()
