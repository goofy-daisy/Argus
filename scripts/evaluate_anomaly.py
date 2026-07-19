#!/usr/bin/env python3
"""
Argus Phase 7 — LSTM Anomaly Detection Evaluation.

Run after training:
    python scripts/evaluate_anomaly.py

Verifies:
  1. Reconstruction MSE distribution on validation set
  2. Zigzag trajectory score > linear trajectory score
  3. Zigzag trajectory score > calibrated threshold (roadmap requirement)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.anomaly.anomaly_detector import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SEQ_LEN     = 30


# ── Synthetic trajectories ────────────────────────────────────────────────────

def make_linear() -> np.ndarray:
    feats = []
    for t in range(SEQ_LEN):
        feats.append([0.1 + 0.015 * t, 0.5, 0.015, 0.0, 0.4])
    return np.array(feats, dtype=np.float32)


def make_zigzag() -> np.ndarray:
    feats, prev_x, prev_y = [], 0.5, 0.5
    for t in range(SEQ_LEN):
        x = 0.5 + 0.35 * np.sin(t * np.pi / 1.5)
        y = 0.5 + 0.01 * t
        feats.append([x, y, x - prev_x, y - prev_y, 0.4])
        prev_x, prev_y = x, y
    return np.array(feats, dtype=np.float32)


# ── MLflow ────────────────────────────────────────────────────────────────────

def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        with mlflow.start_run(tags={"pipeline_stage": "anomaly"}):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in params.items():
                mlflow.log_param(k, v)
        logger.info("Metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("MLflow skipped: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    anom_cfg     = config["anomaly"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    # Read trajectory data directory from config (override hard-coded path)
    DATA_DIR = PROJECT_ROOT / anom_cfg["trajectory_data_dir"]

    model_path = str(PROJECT_ROOT / anom_cfg["model_path"])
    if not Path(model_path).exists():
        logger.error(
            "Weights not found at '%s'. Run scripts/train_anomaly.py first.",
            model_path,
        )
        sys.exit(1)

    detector = AnomalyDetector(
        model_path=model_path,
        device=anom_cfg["device"],
        sequence_length=anom_cfg["sequence_length"],
        feature_dim=anom_cfg["feature_dim"],
        hidden_size=anom_cfg["hidden_size"],
        num_layers=anom_cfg["num_layers"],
    )
    detector.load_model()
    logger.info("Threshold: %.6f", detector.threshold)

    # ── Validation set reconstruction distribution ──
    val_raw    = np.load(str(DATA_DIR / "val_sequences.npy"))
    val_scores = [detector.score(val_raw[i]) for i in range(len(val_raw))]
    val_arr    = np.array(val_scores)
    logger.info(
        "Val MSE — mean=%.6f  std=%.6f  max=%.6f  p95=%.6f",
        val_arr.mean(), val_arr.std(),
        val_arr.max(), np.percentile(val_arr, 95),
    )
    above_threshold = (val_arr > detector.threshold).sum()
    logger.info(
        "Val sequences above threshold: %d / %d  (FP rate: %.2f%%)",
        above_threshold, len(val_arr),
        100 * above_threshold / len(val_arr),
    )

    # ── Synthetic trajectory test ──
    linear_score   = detector.score(make_linear())
    zigzag_score   = detector.score(make_zigzag())
    zigzag_anom, _ = detector.is_anomalous(make_zigzag())
    linear_anom, _ = detector.is_anomalous(make_linear())

    logger.info("─" * 55)
    logger.info("  Linear  score : %.6f  anomalous: %s", linear_score, linear_anom)
    logger.info("  Zigzag  score : %.6f  anomalous: %s", zigzag_score, zigzag_anom)
    logger.info("  Threshold     : %.6f", detector.threshold)
    logger.info("─" * 55)

    passed = True
    if zigzag_score > linear_score:
        logger.info("✅  Zigzag score > Linear score")
    else:
        logger.warning("⚠️   Zigzag score NOT > Linear score")
        passed = False

    if zigzag_anom:
        logger.info("✅  Zigzag classified as anomalous")
    else:
        logger.warning("⚠️   Zigzag NOT classified as anomalous")
        passed = False

    if passed:
        logger.info("✅  Phase 7 roadmap requirement met")
    else:
        logger.warning("⚠️   Phase 7 roadmap requirement not met — retrain with more epochs")

    metrics = {
        "val_mse_mean":     round(float(val_arr.mean()), 6),
        "val_mse_std":      round(float(val_arr.std()),  6),
        "val_fp_rate":      round(float(above_threshold / len(val_arr)), 4),
        "linear_score":     round(linear_score, 6),
        "zigzag_score":     round(zigzag_score, 6),
        "threshold":        round(detector.threshold, 6),
        "zigzag_anomalous": int(zigzag_anom),
    }
    params = {
        "model":       "lstm_autoencoder",
        "seq_len":     anom_cfg["sequence_length"],
        "feat_dim":    anom_cfg["feature_dim"],
        "hidden_size": anom_cfg["hidden_size"],
        "num_layers":  anom_cfg["num_layers"],
    }
    log_to_mlflow(metrics, params, tracking_uri)


if __name__ == "__main__":
    main()
