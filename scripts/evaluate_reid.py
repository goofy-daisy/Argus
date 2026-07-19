#!/usr/bin/env python3
"""
Argus Phase 5 — Re-ID Evaluation on Market-1501.

Computes Rank-1, Rank-5, Rank-10 CMC and mAP against the Market-1501
test gallery and query sets.

Run after training and weight export:
    python scripts/evaluate_reid.py

Target: Rank-1 CMC > 0.87
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.reid.reid import ReIdentifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

MARKET_ROOT = PROJECT_ROOT / "argus" / "data" / "Market-1501" / "Market-1501-v15.09.15"
GALLERY_DIR = MARKET_ROOT / "bounding_box_test"
QUERY_DIR = MARKET_ROOT / "query"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

RANK1_TARGET = 0.87


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def parse_filename(filename: str) -> Tuple[int, int]:
    """Parse Market-1501 filename into (person_id, camera_id).

    Format: {pid:04d}_c{cid}s{seq}_{frame:06d}_{det:02d}.jpg
    Returns (-1, -1) for junk/background images.
    """
    basename = Path(filename).stem
    match = re.match(r"^(\d{4})_c(\d+)s\d+_\d+_\d{2}$", basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def load_images(directory: Path) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Load all valid person images from a directory.

    Skips junk (pid=-1) and background (pid=0) images.

    Returns:
        frames: List of BGR numpy arrays
        pids: numpy int array of person IDs
        cids: numpy int array of camera IDs
    """
    frames, pids, cids = [], [], []
    for img_path in sorted(directory.glob("*.jpg")):
        pid, cid = parse_filename(img_path.name)
        if pid <= 0:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frames.append(frame)
        pids.append(pid)
        cids.append(cid)
    return frames, np.array(pids), np.array(cids)


def extract_all_embeddings(
    reid: ReIdentifier,
    frames: List[np.ndarray],
    label: str,
) -> np.ndarray:
    """Extract embeddings for all images in a list using full-frame crop."""
    embeddings = []
    total = len(frames)
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        emb = reid.extract_embedding(frame, (0.0, 0.0, float(w), float(h)))
        embeddings.append(emb)
        if (i + 1) % 500 == 0:
            logger.info("  %s: %d / %d", label, i + 1, total)
    logger.info("  %s: %d / %d done", label, total, total)
    return np.array(embeddings, dtype=np.float32)


def compute_cmc_map(
    query_embeddings: np.ndarray,
    query_pids: np.ndarray,
    query_cids: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_cids: np.ndarray,
    max_rank: int = 10,
) -> Tuple[np.ndarray, float]:
    """Compute CMC curve and mAP.

    Follows the standard Market-1501 evaluation protocol:
    - Same-camera same-person gallery images are excluded from ranking.
    - CMC[k] = fraction of queries where correct match is in top-k.
    - mAP = mean average precision across all queries.

    Returns:
        cmc: numpy array of shape (max_rank,), cumulative match curve
        mAP: float, mean average precision
    """
    num_queries = len(query_pids)
    cmc = np.zeros(max_rank, dtype=float)
    ap_sum = 0.0

    for q_idx in range(num_queries):
        q_emb = query_embeddings[q_idx]
        q_pid = query_pids[q_idx]
        q_cid = query_cids[q_idx]

        # Cosine similarity — embeddings are L2-normalised, dot = cosine
        scores = gallery_embeddings @ q_emb

        # Sort descending
        order = np.argsort(-scores)
        sorted_pids = gallery_pids[order]
        sorted_cids = gallery_cids[order]

        # Exclude junk: same person, same camera
        junk_mask = (sorted_pids == q_pid) & (sorted_cids == q_cid)
        valid_mask = ~junk_mask
        valid_pids = sorted_pids[valid_mask]
        matches = (valid_pids == q_pid).astype(float)

        # CMC: find rank of first correct match
        for k in range(min(max_rank, len(matches))):
            if matches[k] > 0:
                cmc[k:] += 1
                break

        # AP for this query
        num_rel = matches.sum()
        if num_rel > 0:
            precisions = np.cumsum(matches) / (np.arange(len(matches)) + 1)
            ap_sum += (precisions * matches).sum() / num_rel

    cmc = cmc / num_queries
    m_ap = ap_sum / num_queries
    return cmc, m_ap


def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        with mlflow.start_run(tags={"pipeline_stage": "reid", "dataset": "market1501"}):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in params.items():
                mlflow.log_param(k, v)
        logger.info("Metrics logged to MLflow at %s", tracking_uri)
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    reid_cfg = config["reid"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    model_path = str(PROJECT_ROOT / reid_cfg["model_path"])  # config-driven: reid.model_path in config.yaml
    if not Path(model_path).exists():
        logger.error(
            "Weights not found at '%s'. "
            "Run scripts/train_reid.py then scripts/export_reid_weights.py first.",
            model_path,
        )
        sys.exit(1)

    reid = ReIdentifier(
        model_path=model_path,
        device=reid_cfg["device"],
        similarity_threshold=reid_cfg["similarity_threshold"],
        input_height=reid_cfg["input_height"],
        input_width=reid_cfg["input_width"],
        num_train_pids=reid_cfg["num_train_pids"],
    )

    logger.info("Loading OSNet-x1.0...")
    reid.load_model()

    logger.info("Loading gallery images from %s...", GALLERY_DIR)
    gal_frames, gal_pids, gal_cids = load_images(GALLERY_DIR)
    logger.info("Gallery: %d images, %d unique pids", len(gal_frames), len(np.unique(gal_pids)))

    logger.info("Loading query images from %s...", QUERY_DIR)
    qry_frames, qry_pids, qry_cids = load_images(QUERY_DIR)
    logger.info("Query:   %d images, %d unique pids", len(qry_frames), len(np.unique(qry_pids)))

    logger.info("Extracting gallery embeddings...")
    start = time.perf_counter()
    gal_embeddings = extract_all_embeddings(reid, gal_frames, "gallery")
    logger.info("Gallery embeddings done in %.1f s", time.perf_counter() - start)

    logger.info("Extracting query embeddings...")
    start = time.perf_counter()
    qry_embeddings = extract_all_embeddings(reid, qry_frames, "query")
    logger.info("Query embeddings done in %.1f s", time.perf_counter() - start)

    logger.info("Computing CMC and mAP...")
    cmc, m_ap = compute_cmc_map(
        qry_embeddings, qry_pids, qry_cids,
        gal_embeddings, gal_pids, gal_cids,
        max_rank=10,
    )

    logger.info("─" * 55)
    logger.info("  Rank-1  :  %.4f  (target > %.2f)", cmc[0], RANK1_TARGET)
    logger.info("  Rank-5  :  %.4f", cmc[4])
    logger.info("  Rank-10 :  %.4f", cmc[9])
    logger.info("  mAP     :  %.4f", m_ap)
    logger.info("─" * 55)

    if cmc[0] >= RANK1_TARGET:
        logger.info("✅  Rank-1 target met: %.4f >= %.2f", cmc[0], RANK1_TARGET)
    else:
        logger.warning("⚠️   Rank-1 below target: %.4f < %.2f", cmc[0], RANK1_TARGET)

    metrics = {
        "rank1": round(float(cmc[0]), 4),
        "rank5": round(float(cmc[4]), 4),
        "rank10": round(float(cmc[9]), 4),
        "mAP": round(float(m_ap), 4),
    }
    params = {
        "model": "osnet_x1_0",
        "dataset": "market1501",
        "similarity_threshold": reid_cfg["similarity_threshold"],
        "input_height": reid_cfg["input_height"],
        "input_width": reid_cfg["input_width"],
    }
    log_to_mlflow(metrics, params, tracking_uri)


if __name__ == "__main__":
    main()
