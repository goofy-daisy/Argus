#!/usr/bin/env python3
"""
Argus Phase 4 — Detection and Tracking Evaluation on MOT17-02-FRCNN.

Runs YOLOv8m detection + DeepSORT tracking on all 600 frames of the
MOT17-02-FRCNN training sequence. Computes MOTA and IDF1 using motmetrics
against the MOT17 ground truth annotations.

Targets: MOTA > 0.70, IDF1 > 0.65

Logs results to MLflow (argus experiment, tagged pipeline_stage=detection_tracking).

Run from project root with venv active:
    python scripts/evaluate_mot17.py
"""
from __future__ import annotations

import configparser
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
# NumPy 2.0 removed np.asfarray — patch immediately after numpy import,
# before any other module that might trigger a motmetrics import transitively.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.detection.detector import Detector
from argus.ingestion.video_reader import VideoReader
from argus.tracking.tracker import Tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SEQ_DIR = PROJECT_ROOT / "argus" / "data" / "MOT17" / "train" / "MOT17-02-FRCNN"
IMG_DIR = SEQ_DIR / "img1"
GT_FILE = SEQ_DIR / "gt" / "gt.txt"
SEQ_INFO = SEQ_DIR / "seqinfo.ini"
MODEL_PATH = str(PROJECT_ROOT / "argus" / "models" / "yolov8m.pt")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ── Targets ───────────────────────────────────────────────────────────────────

MOTA_TARGET = 0.70
IDF1_TARGET = 0.65



# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def load_seq_info() -> dict:
    if not SEQ_INFO.exists():
        raise FileNotFoundError(
            f"\nseqinfo.ini not found at:\n  {SEQ_INFO}\n"
            f"Ensure MOT17-02-FRCNN is placed at:\n"
            f"  argus/data/MOT17/train/MOT17-02-FRCNN/"
        )
    parser = configparser.ConfigParser()
    parser.read(str(SEQ_INFO))
    return {
        "seq_length": int(parser["Sequence"]["seqLength"]),
        "frame_rate": int(parser["Sequence"]["frameRate"]),
        "im_width": int(parser["Sequence"]["imWidth"]),
        "im_height": int(parser["Sequence"]["imHeight"]),
    }


def load_ground_truth() -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Load and filter MOT17 ground truth.

    Filters to class=1 (pedestrian) and conf=1 (non-distractor only).

    Returns:
        Dict mapping frame_id (1-indexed) to list of
        (track_id, x, y, w, h) tuples.
    """
    gt = pd.read_csv(
        str(GT_FILE),
        header=None,
        names=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"],
    )
    gt = gt[(gt["conf"] == 1) & (gt["class"] == 1)].copy()

    gt_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for _, row in gt.iterrows():
        gt_by_frame[int(row["frame"])].append(
            (int(row["id"]), float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
        )
    logger.info(
        "Ground truth loaded: %d frames, %d total annotations",
        len(gt_by_frame),
        len(gt),
    )
    return gt_by_frame


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to (x, y, w, h) for motmetrics."""
    return x1, y1, x2 - x1, y2 - y1


def compute_metrics(
    gt_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]],
    pred_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]],
    total_frames: int,
) -> dict:
    """Compute MOTA and IDF1 using motmetrics."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in range(1, total_frames + 1):
        gt_entries = gt_by_frame.get(frame_id, [])
        pred_entries = pred_by_frame.get(frame_id, [])

        gt_ids = [e[0] for e in gt_entries]
        gt_boxes = np.array([[e[1], e[2], e[3], e[4]] for e in gt_entries], dtype=float) if gt_entries else np.empty((0, 4))

        pred_ids = [e[0] for e in pred_entries]
        pred_boxes = np.array([[e[1], e[2], e[3], e[4]] for e in pred_entries], dtype=float) if pred_entries else np.empty((0, 4))

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            acc.update([], [], np.empty((0, 0)))
            continue

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "num_frames"],
        name="MOT17-02-FRCNN",
    )

    return {
        "mota": float(summary["mota"].iloc[0]),
        "idf1": float(summary["idf1"].iloc[0]),
        "num_switches": int(summary["num_switches"].iloc[0]),
        "num_frames": int(summary["num_frames"].iloc[0]),
    }


def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    """Log evaluation metrics to MLflow. Skips gracefully if unreachable."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")

        with mlflow.start_run(
            tags={
                "pipeline_stage": "detection_tracking",
                "sequence": "MOT17-02-FRCNN",
            }
        ):
            mlflow.log_metric("mota", metrics["mota"])
            mlflow.log_metric("idf1", metrics["idf1"])
            mlflow.log_metric("num_switches", metrics["num_switches"])
            mlflow.log_metric("eval_fps", metrics.get("eval_fps", 0.0))
            for k, v in params.items():
                mlflow.log_param(k, v)

        logger.info("Metrics logged to MLflow at %s", tracking_uri)
    except Exception as exc:
        logger.warning("MLflow logging skipped — server may not be running: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    tracking_uri = config["mlflow"]["tracking_uri"]
    seq_info = load_seq_info()
    total_frames = seq_info["seq_length"]

    logger.info("Sequence: MOT17-02-FRCNN  |  frames=%d  |  fps=%d",
                total_frames, seq_info["frame_rate"])
    logger.info("Detection  — conf=%.2f | nms_iou=%.2f | imgsz=%d",
                config["detection"]["confidence_threshold"],
                config["detection"]["nms_iou_threshold"],
                config["detection"]["imgsz"])
    logger.info("Tracking   — track_thresh=%.2f | track_buffer=%d",
                config["tracking"]["track_thresh"],
                config["tracking"]["track_buffer"])

    detector = Detector(
        model_path=MODEL_PATH,
        config=config,
    )
    tracker = Tracker(config=config)

    logger.info("Loading YOLOv8m...")
    detector.load_model()
    logger.info("Models ready. Starting evaluation...")

    gt_by_frame = load_ground_truth()
    pred_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)

    frame_id = 0
    start = time.perf_counter()

    with VideoReader(str(IMG_DIR), stride=1) as reader:
        for frame in reader.read_frames():
            frame_id += 1

            # Raw uint8 BGR frame passed directly — YOLOv8 handles preprocessing internally.
            # Preprocessor pipeline is used by downstream modules (ReID, action, anomaly) only.
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)

            for track in tracks:
                x1, y1, x2, y2 = track.to_ltrb()
                x, y, w, h = xyxy_to_xywh(x1, y1, x2, y2)
                pred_by_frame[frame_id].append((track.track_id, x, y, w, h))

            if frame_id % 100 == 0:
                elapsed = time.perf_counter() - start
                fps = frame_id / elapsed
                logger.info("  Frame %d / %d  (%.1f fps)", frame_id, total_frames, fps)

    elapsed = time.perf_counter() - start
    eval_fps = frame_id / elapsed if elapsed > 0 else 0.0
    logger.info("Pipeline complete — %d frames in %.1f s (%.1f fps)", frame_id, elapsed, eval_fps)

    logger.info("Computing MOTA / IDF1...")
    metrics = compute_metrics(gt_by_frame, pred_by_frame, total_frames)
    metrics["eval_fps"] = round(eval_fps, 2)

    logger.info("─" * 55)
    logger.info("  MOTA          :  %.4f  (target > %.2f)", metrics["mota"], MOTA_TARGET)
    logger.info("  IDF1          :  %.4f  (target > %.2f)", metrics["idf1"], IDF1_TARGET)
    logger.info("  ID Switches   :  %d", metrics["num_switches"])
    logger.info("  Eval FPS      :  %.2f", metrics["eval_fps"])
    logger.info("─" * 55)

    if metrics["mota"] >= MOTA_TARGET:
        logger.info("✅  MOTA target met:  %.4f >= %.2f", metrics["mota"], MOTA_TARGET)
    else:
        logger.warning("⚠️   MOTA below target: %.4f < %.2f", metrics["mota"], MOTA_TARGET)

    if metrics["idf1"] >= IDF1_TARGET:
        logger.info("✅  IDF1 target met:  %.4f >= %.2f", metrics["idf1"], IDF1_TARGET)
    else:
        logger.warning("⚠️   IDF1 below target: %.4f < %.2f", metrics["idf1"], IDF1_TARGET)

    params = {
        "confidence_threshold": config["detection"]["confidence_threshold"],
        "nms_iou_threshold": config["detection"]["nms_iou_threshold"],
        "imgsz": config["detection"]["imgsz"],
        "track_thresh": config["tracking"]["track_thresh"],
        "track_buffer": config["tracking"]["track_buffer"],
        "match_thresh": config["tracking"]["match_thresh"],
        "sequence": "MOT17-02-FRCNN",
        "total_frames": total_frames,
    }
    log_to_mlflow(metrics, params, tracking_uri)


if __name__ == "__main__":
    main()