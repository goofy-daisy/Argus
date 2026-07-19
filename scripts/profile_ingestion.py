#!/usr/bin/env python3
"""
Argus Phase 3 — Ingestion Pipeline Throughput Profiler.

Runs the full VideoReader → Preprocessor → FrameBatcher pipeline and
measures throughput. Uses MOT17-02 image sequence if present, otherwise
falls back to a 300-frame 1080p synthetic video.

Logs fps, frame count, and dropped count to MLflow (argus experiment,
tagged pipeline_stage=ingestion). Target: > 25 fps on Apple Silicon.

Run from project root with venv active:
    python scripts/profile_ingestion.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.ingestion.frame_batcher import FrameBatcher
from argus.ingestion.preprocessor import Preprocessor
from argus.ingestion.video_reader import VideoReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MOT17_SEQ = (
    PROJECT_ROOT
    / "argus"
    / "data"
    / "MOT17"
    / "train"
    / "MOT17-02-FRCNN"
    / "img1"
)
SYNTHETIC_FRAMES = 300
SYNTHETIC_WIDTH = 1920
SYNTHETIC_HEIGHT = 1080
TARGET_FPS = 25.0
PROFILE_STRIDE = 3
PROFILE_BATCH_SIZE = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def make_synthetic_video(path: Path, num_frames: int, width: int, height: int) -> None:
    """Write a synthetic MP4 video filled with random noise frames."""
    logger.info(
        "Generating synthetic video: %d frames at %dx%d → %s",
        num_frames,
        width,
        height,
        path,
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    logger.info("Synthetic video written.")


def run_pipeline(source: str, stride: int, batch_size: int) -> dict:
    """Run the full ingestion pipeline and return timing metrics."""
    preprocessor = Preprocessor()
    batcher = FrameBatcher(batch_size=batch_size)

    total_frames = 0
    total_batches = 0
    start = time.perf_counter()

    with VideoReader(source, stride=stride) as reader:
        for batch in batcher.batch(reader.read_frames()):
            for frame in batch:
                _ = preprocessor.preprocess_rgb(frame)
            total_frames += len(batch)
            total_batches += 1
        dropped = reader.dropped_count

    elapsed = time.perf_counter() - start
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "dropped_frames": dropped,
        "total_batches": total_batches,
        "elapsed_seconds": round(elapsed, 3),
        "stride": stride,
        "batch_size": batch_size,
    }


def log_to_mlflow(metrics: dict, source_type: str, tracking_uri: str) -> None:
    """Log metrics to MLflow. Skips gracefully if the server is unreachable."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")

        with mlflow.start_run(
            tags={"pipeline_stage": "ingestion", "source_type": source_type}
        ):
            mlflow.log_metric("fps", metrics["fps"])
            mlflow.log_metric("total_frames", metrics["total_frames"])
            mlflow.log_metric("dropped_frames", metrics["dropped_frames"])
            mlflow.log_metric("elapsed_seconds", metrics["elapsed_seconds"])
            mlflow.log_param("stride", metrics["stride"])
            mlflow.log_param("batch_size", metrics["batch_size"])
            mlflow.log_param("source_type", source_type)

        logger.info("Metrics logged to MLflow at %s", tracking_uri)
    except Exception as exc:
        logger.warning(
            "MLflow logging skipped — server may not be running: %s", exc
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    tracking_uri = config["mlflow"]["tracking_uri"]

    # Determine source
    if MOT17_SEQ.is_dir() and any(MOT17_SEQ.iterdir()):
        source = str(MOT17_SEQ)
        source_type = "mot17_image_sequence"
        logger.info("Source: MOT17-02 image sequence at %s", source)
    else:
        logger.info(
            "MOT17-02 not found — using synthetic %dx%d video (%d frames)",
            SYNTHETIC_WIDTH,
            SYNTHETIC_HEIGHT,
            SYNTHETIC_FRAMES,
        )
        tmp_dir = Path(tempfile.mkdtemp())
        video_path = tmp_dir / "synthetic_argus_phase3.mp4"
        make_synthetic_video(video_path, SYNTHETIC_FRAMES, SYNTHETIC_WIDTH, SYNTHETIC_HEIGHT)
        source = str(video_path)
        source_type = "synthetic"

    logger.info(
        "Running pipeline — stride=%d, batch_size=%d ...",
        PROFILE_STRIDE,
        PROFILE_BATCH_SIZE,
    )
    metrics = run_pipeline(source, PROFILE_STRIDE, PROFILE_BATCH_SIZE)

    # ── Results ──
    logger.info("─" * 52)
    logger.info("  Throughput :  %.2f fps", metrics["fps"])
    logger.info("  Frames     :  %d processed, %d dropped", metrics["total_frames"], metrics["dropped_frames"])
    logger.info("  Batches    :  %d", metrics["total_batches"])
    logger.info("  Elapsed    :  %.3f s", metrics["elapsed_seconds"])
    logger.info("  Source     :  %s", source_type)
    logger.info("─" * 52)

    if metrics["fps"] >= TARGET_FPS:
        logger.info(
            "✅  Throughput target met: %.2f fps >= %.1f fps", metrics["fps"], TARGET_FPS
        )
    else:
        logger.warning(
            "⚠️   Throughput below target: %.2f fps < %.1f fps — investigate bottleneck",
            metrics["fps"],
            TARGET_FPS,
        )

    log_to_mlflow(metrics, source_type, tracking_uri)


if __name__ == "__main__":
    main()
