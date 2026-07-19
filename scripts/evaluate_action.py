#!/usr/bin/env python3
"""
Argus Phase 6 — X3D-S Action Recognition Evaluation on UCF101 Proxy Test Set.

Run after training and weight export:
    python scripts/evaluate_action.py

Target: Top-1 accuracy > 0.80 on UCF101 proxy test set.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.action.action_classifier import ActionClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

UCF101_ROOT   = PROJECT_ROOT / "argus" / "data" / "UCF101"
CONFIG_PATH   = PROJECT_ROOT / "config.yaml"
CLIPS_PER_VID = 5
TOP1_TARGET   = 0.80

PROXY_MAP = {
    "normal":           ["TaiChi", "WalkingWithDog"],
    "loitering":        ["Knitting", "ShavingBeard"],
    "running":          ["Basketball", "SoccerPenalty"],
    "falling":          ["FloorGymnastics", "LongJump"],
    "crowd_formation":  ["BandMarching", "MilitaryParade"],
}
ARGUS_LABELS = ["normal", "loitering", "running", "falling", "crowd_formation"]


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def load_test_samples():
    ucf_to_argus = {}
    for argus_label, ucf_classes in PROXY_MAP.items():
        idx = ARGUS_LABELS.index(argus_label)
        for uc in ucf_classes:
            ucf_to_argus[uc] = (argus_label, idx)

    test_file = UCF101_ROOT / "ucfTrainTestlist" / "testlist01.txt"
    samples = []
    with open(test_file) as fh:
        for line in fh:
            video_rel = line.strip().split()[0]
            class_name = video_rel.split("/")[0]
            if class_name in ucf_to_argus:
                vpath = UCF101_ROOT / "UCF-101" / video_rel
                if vpath.exists():
                    label_str, label_idx = ucf_to_argus[class_name]
                    samples.append((str(vpath), label_str, label_idx))
    return samples


def extract_clips(video_path: str, classifier: ActionClassifier, n_clips: int = 5):
    """Extract n_clips uniformly spaced 16-frame clips from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        total = 16

    clip_starts = np.linspace(0, max(0, total - 16), n_clips, dtype=int)
    clips = []
    for start in clip_starts:
        cap = cv2.VideoCapture(video_path)
        frames = []
        for fi in range(start, start + 16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(fi, total - 1)))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(
                    np.zeros(
                        (classifier.clip_size, classifier.clip_size, 3),
                        dtype=np.uint8,
                    )
                )
        cap.release()
        while len(frames) < 16:
            frames.append(frames[-1].copy())
        h, w = frames[0].shape[:2]
        clip_arr = classifier.extract_clip(frames, (0.0, 0.0, float(w), float(h)))
        clips.append(clip_arr)
    return clips


def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        with mlflow.start_run(tags={"pipeline_stage": "action", "dataset": "ucf101_proxy"}):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for k, v in params.items():
                mlflow.log_param(k, v)
        logger.info("Metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    config = load_config()
    action_cfg = config["action"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    model_path = str(PROJECT_ROOT / action_cfg["model_path"])
    if not Path(model_path).exists():
        logger.error(
            "Weights not found at '%s'. "
            "Run scripts/train_action.py then scripts/export_action_weights.py.",
            model_path,
        )
        sys.exit(1)

    classifier = ActionClassifier(
        model_path=model_path,
        device=action_cfg["device"],
        num_classes=action_cfg["num_classes"],
        clip_frames=action_cfg["clip_frames"],
        clip_size=action_cfg["clip_size"],
        video_mean=action_cfg["video_mean"],
        video_std=action_cfg["video_std"],
    )
    logger.info("Loading X3D-S...")
    classifier.load_model()

    samples = load_test_samples()
    logger.info("Test videos: %d", len(samples))

    correct = 0
    per_class_correct = {l: 0 for l in ARGUS_LABELS}
    per_class_total   = {l: 0 for l in ARGUS_LABELS}

    start = time.perf_counter()
    for i, (vpath, gt_label, gt_idx) in enumerate(samples):
        clips = extract_clips(vpath, classifier, n_clips=CLIPS_PER_VID)
        predictions = []
        for clip in clips:
            pred_label, _ = classifier.classify(clip)
            predictions.append(pred_label)

        # Majority vote
        voted_label = Counter(predictions).most_common(1)[0][0]
        per_class_total[gt_label] += 1
        if voted_label == gt_label:
            correct += 1
            per_class_correct[gt_label] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - start
            logger.info("  %d / %d  (%.1f s)", i + 1, len(samples), elapsed)

    elapsed = time.perf_counter() - start
    overall_top1 = correct / len(samples) if samples else 0.0

    logger.info("─" * 55)
    logger.info("  Overall Top-1 : %.4f  (target > %.2f)", overall_top1, TOP1_TARGET)
    logger.info("  Videos tested : %d  (%.1f s)", len(samples), elapsed)
    logger.info("  Per-class accuracy:")
    for label in ARGUS_LABELS:
        n = per_class_total[label]
        acc = per_class_correct[label] / n if n > 0 else 0.0
        logger.info("    %-20s  %.4f  (%d videos)", label, acc, n)
    logger.info("─" * 55)

    if overall_top1 >= TOP1_TARGET:
        logger.info("✅  Top-1 target met: %.4f >= %.2f", overall_top1, TOP1_TARGET)
    else:
        logger.warning("⚠️   Top-1 below target: %.4f < %.2f", overall_top1, TOP1_TARGET)

    metrics = {
        "top1_overall": round(overall_top1, 4),
        **{
            f"top1_{l}": round(
                per_class_correct[l] / per_class_total[l], 4
            ) if per_class_total[l] > 0 else 0.0
            for l in ARGUS_LABELS
        },
    }
    params = {
        "model": "x3d_s",
        "clips_per_video": CLIPS_PER_VID,
        "majority_vote": True,
        "num_classes": action_cfg["num_classes"],
        "clip_frames": action_cfg["clip_frames"],
        "clip_size": action_cfg["clip_size"],
    }
    log_to_mlflow(metrics, params, tracking_uri)


if __name__ == "__main__":
    main()
