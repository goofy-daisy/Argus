#!/usr/bin/env python3
"""
Argus Phase 4 — YOLOv8m Fine-Tuning on MOT17.

DO NOT RUN this script from Claude Code. Trigger manually when ready:
    python scripts/train_detector.py

Training time: approximately 30–60 minutes on Apple Silicon MPS.
Output: argus/models/yolov8m_finetuned.pt

Prerequisites (all already satisfied if data preparation is complete):
    argus/data/MOT17/mot17_yolo.yaml
    argus/data/MOT17/train_images.txt      (4715 images)
    argus/data/MOT17/val_images.txt        (599 images)
    argus/data/MOT17/yolo_labels/          (7 FRCNN subdirs with YOLO .txt labels)
    argus/models/yolov8m.pt               (COCO pretrained)

After training completes:
    python scripts/evaluate_mot17.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_YAML      = PROJECT_ROOT / "argus" / "data" / "MOT17" / "mot17_yolo.yaml"
TRAIN_LIST     = PROJECT_ROOT / "argus" / "data" / "MOT17" / "train_images.txt"
VAL_LIST       = PROJECT_ROOT / "argus" / "data" / "MOT17" / "val_images.txt"
YOLO_LABELS    = PROJECT_ROOT / "argus" / "data" / "MOT17" / "yolo_labels"
BASE_WEIGHTS   = PROJECT_ROOT / "argus" / "models" / "yolov8m.pt"
OUTPUT_WEIGHTS = PROJECT_ROOT / "argus" / "models" / "yolov8m_finetuned.pt"
RUN_DIR        = PROJECT_ROOT / "runs" / "detect"
CONFIG_PATH    = PROJECT_ROOT / "config.yaml"
# MLflow server: http://localhost:5001  (macOS AirPlay occupies 5000)
MLFLOW_URI_DEFAULT = "http://localhost:5001"

# ── Training hyperparameters ───────────────────────────────────────────────────

EPOCHS        = 1
IMGSZ         = 416
BATCH         = 4
DEVICE        = "mps"
LR0           = 0.01
LRF           = 0.01
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3
CLOSE_MOSAIC  = 3
RUN_NAME      = "yolov8m_mot17_ft"
LAST_CKPT     = RUN_DIR / RUN_NAME / "weights" / "last.pt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_prerequisites() -> None:
    """Verify all required data files and base weights exist."""
    required = [DATA_YAML, TRAIN_LIST, VAL_LIST, YOLO_LABELS, BASE_WEIGHTS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.error("Missing required files / directories:")
        for p in missing:
            logger.error("  %s", p)
        logger.error(
            "Ensure scripts/prepare_mot17_trajectories.py has been run "
            "and yolov8m.pt is present at argus/models/."
        )
        sys.exit(1)
    logger.info("All prerequisites verified.")


def read_final_metrics(results_csv: Path) -> dict:
    """Read the last row of ultralytics results.csv and return key metrics."""
    try:
        import pandas as pd

        df = pd.read_csv(str(results_csv))
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        col_map = {
            "metrics/mAP50(B)":    "mAP50",
            "metrics/mAP50-95(B)": "mAP50-95",
            "train/box_loss":      "train_box_loss",
            "train/cls_loss":      "train_cls_loss",
            "train/dfl_loss":      "train_dfl_loss",
        }
        return {
            metric_name: float(last[csv_col])
            for csv_col, metric_name in col_map.items()
            if csv_col in last.index
        }
    except Exception as exc:
        logger.warning("Could not read results.csv: %s", exc)
        return {}


def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        with mlflow.start_run(
            tags={
                "pipeline_stage": "detection_training",
                "sequence":       "MOT17-02-FRCNN",
            }
        ):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
        logger.info("Metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    check_prerequisites()

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    tracking_uri = config["mlflow"]["tracking_uri"]

    logger.info("=" * 55)
    logger.info("  YOLOv8m MOT17 Fine-Tuning")
    logger.info("  Base weights : %s", BASE_WEIGHTS)
    logger.info("  Data config  : %s", DATA_YAML)
    logger.info("  Device       : %s", DEVICE)
    logger.info("  Epochs       : %d  |  Batch: %d  |  imgsz: %d", EPOCHS, BATCH, IMGSZ)
    logger.info("=" * 55)

    from ultralytics import YOLO

    if LAST_CKPT.exists():
        logger.info("Resuming from checkpoint: %s", LAST_CKPT)
        model = YOLO(str(LAST_CKPT))
        model.train(resume=True)
    else:
        model = YOLO(str(BASE_WEIGHTS))
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            lr0=LR0,
            lrf=LRF,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            warmup_epochs=WARMUP_EPOCHS,
            close_mosaic=CLOSE_MOSAIC,
            project=str(RUN_DIR),
            name=RUN_NAME,
            exist_ok=True,
        )

    # ── Copy best weights to canonical output path ─────────────────────────────
    best_pt = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    if best_pt.exists():
        OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_pt), str(OUTPUT_WEIGHTS))
        logger.info("Best weights saved to %s", OUTPUT_WEIGHTS)
    else:
        logger.warning("best.pt not found at expected path %s", best_pt)

    # ── Read and log final metrics ────────────────────────────────────────────
    results_csv = RUN_DIR / RUN_NAME / "results.csv"
    final_metrics = read_final_metrics(results_csv)
    if final_metrics:
        logger.info("─" * 55)
        for k, v in final_metrics.items():
            logger.info("  %-20s : %.4f", k, v)
        logger.info("─" * 55)

    params = {
        "base_model":        "yolov8m",
        "dataset":           "MOT17-FRCNN",
        "epochs":            EPOCHS,
        "imgsz":             IMGSZ,
        "batch":             BATCH,
        "device":            DEVICE,
        "lr0":               LR0,
        "lrf":               LRF,
        "warmup_epochs":     WARMUP_EPOCHS,
        "close_mosaic":      CLOSE_MOSAIC,
        "output_model_path": str(OUTPUT_WEIGHTS),
    }
    log_to_mlflow(final_metrics, params, tracking_uri)

    logger.info("Training complete.")
    logger.info("Fine-tuned weights: %s", OUTPUT_WEIGHTS)
    logger.info("Next step: python scripts/evaluate_mot17.py")


if __name__ == "__main__":
    main()
