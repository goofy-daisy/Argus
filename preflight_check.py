#!/usr/bin/env python3
"""
Argus Pre-Flight Check
Run from project root with venv active before starting the training pipeline.
Verifies all data paths, script syntax, imports, config values, and critical
code consistency points. Fix every failure before running run_all.py.

Usage:
    python preflight_check.py
"""
import os
import sys
import importlib
import py_compile
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS_COUNT = 0
FAIL_COUNT = 0

def ok(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  ✅  {msg}")

def fail(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  ❌  {msg}")

def check_path(rel, label):
    if (ROOT / rel).exists():
        ok(label)
    else:
        fail(f"MISSING — {label}  [{rel}]")

def check_dir_nonempty(rel, label, glob="*"):
    p = ROOT / rel
    if not p.exists():
        fail(f"MISSING DIR — {label}  [{rel}]")
        return
    files = list(p.glob(glob))
    if files:
        ok(f"{label}  ({len(files)} files)")
    else:
        fail(f"EMPTY DIR — {label}  [{rel}]")


# ══════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════")
print("  ARGUS PRE-FLIGHT CHECK")
print("══════════════════════════════════════════════════════\n")

# ── SECTION 1: DATA PATHS ─────────────────────────────────────
print("── DATA PATHS ──────────────────────────────────────────")

check_path("argus/data/MOT17/mot17_yolo.yaml",                    "MOT17 YOLO config")
check_path("argus/data/MOT17/train_images.txt",                   "MOT17 train image list")
check_path("argus/data/MOT17/val_images.txt",                     "MOT17 val image list")
check_dir_nonempty("argus/data/MOT17/yolo_labels",                "MOT17 YOLO labels root")
check_path("argus/data/MOT17/train/MOT17-02-FRCNN/gt/gt.txt",     "MOT17-02 ground truth")
check_path("argus/data/MOT17/train/MOT17-02-FRCNN/seqinfo.ini",   "MOT17-02 seqinfo.ini")

check_dir_nonempty("argus/data/Market-1501/Market-1501-v15.09.15/bounding_box_train",
                   "Market-1501 bounding_box_train", "*.jpg")
check_dir_nonempty("argus/data/Market-1501/Market-1501-v15.09.15/bounding_box_test",
                   "Market-1501 bounding_box_test",  "*.jpg")
check_dir_nonempty("argus/data/Market-1501/Market-1501-v15.09.15/query",
                   "Market-1501 query",              "*.jpg")

check_dir_nonempty("argus/data/UCF101/UCF-101",                   "UCF101 video root")
check_path("argus/data/UCF101/ucfTrainTestlist/trainlist01.txt",  "UCF101 train split")
check_path("argus/data/UCF101/ucfTrainTestlist/testlist01.txt",   "UCF101 test split")
check_path("argus/data/UCF101/ucfTrainTestlist/classInd.txt",     "UCF101 class index")

check_dir_nonempty("argus/data/FLIR_ADAS/FLIR_ADAS_v2/images_rgb_train/data",
                   "FLIR RGB train",     "*.jpg")
check_dir_nonempty("argus/data/FLIR_ADAS/FLIR_ADAS_v2/images_thermal_train/data",
                   "FLIR thermal train", "*.jpg")
check_dir_nonempty("argus/data/FLIR_ADAS/FLIR_ADAS_v2/images_rgb_val/data",
                   "FLIR RGB val",       "*.jpg")
check_dir_nonempty("argus/data/FLIR_ADAS/FLIR_ADAS_v2/images_thermal_val/data",
                   "FLIR thermal val",   "*.jpg")

check_path("argus/data/trajectories/train_sequences.npy", "Trajectory train sequences")
check_path("argus/data/trajectories/val_sequences.npy",   "Trajectory val sequences")
check_path("argus/data/trajectories/metadata.json",       "Trajectory metadata")

check_path("argus/models/yolov8m.pt", "YOLOv8m base weights")


# ── SECTION 2: UCF101 PROXY CLASS DIRECTORIES ─────────────────
print("\n── UCF101 PROXY CLASSES ────────────────────────────────")
proxy_classes = [
    "TaiChi", "WalkingWithDog",
    "Knitting", "ShavingBeard",
    "Basketball", "SoccerPenalty",
    "Diving", "HighJump",
    "BandMarching", "MilitaryParade",
]
for cls in proxy_classes:
    p = ROOT / "argus/data/UCF101/UCF-101" / cls
    if p.exists():
        n = len(list(p.glob("*.avi")))
        if n > 0:
            ok(f"{cls}  ({n} videos)")
        else:
            fail(f"{cls}  — directory exists but contains no .avi files")
    else:
        fail(f"{cls}  — directory not found")


# ── SECTION 3: SCRIPT SYNTAX ──────────────────────────────────
print("\n── SCRIPT SYNTAX ───────────────────────────────────────")
scripts_to_check = [
    "scripts/train_detector.py",
    "scripts/train_reid.py",
    "scripts/train_action.py",
    "scripts/train_anomaly.py",
    "scripts/train_fusion.py",
    "scripts/export_reid_weights.py",
    "scripts/export_action_weights.py",
    "scripts/evaluate_mot17.py",
    "scripts/evaluate_reid.py",
    "scripts/evaluate_action.py",
    "scripts/evaluate_anomaly.py",
    "scripts/calibrate_and_evaluate.py",
]
for s in scripts_to_check:
    try:
        py_compile.compile(str(ROOT / s), doraise=True)
        ok(s.split("/")[-1])
    except py_compile.PyCompileError as e:
        fail(f"{s.split('/')[-1]}  — {e}")


# ── SECTION 4: KEY IMPORTS ────────────────────────────────────
print("\n── KEY IMPORTS ─────────────────────────────────────────")
import_checks = [
    ("torch",                                    "PyTorch"),
    ("torchreid",                                "torchreid"),
    ("ultralytics",                              "ultralytics"),
    ("ultralytics.trackers.byte_tracker",        "BYTETracker (ultralytics)"),
    ("cv2",                                      "OpenCV"),
    ("motmetrics",                               "motmetrics"),
    ("mlflow",                                   "MLflow"),
    ("pandas",                                   "pandas"),
    ("scipy",                                    "scipy"),
]
for mod_name, label in import_checks:
    try:
        importlib.import_module(mod_name)
        ok(label)
    except Exception as e:
        fail(f"{label}  — {e}")

try:
    import torch
    if torch.backends.mps.is_available():
        ok("MPS available (Apple Silicon GPU)")
    else:
        fail("MPS not available — training will run on CPU only (very slow)")
except Exception as e:
    fail(f"MPS check failed  — {e}")

try:
    from argus.tracking.tracker import Tracker
    ok("argus.tracking.tracker (ByteTrack)")
except Exception as e:
    fail(f"argus.tracking.tracker  — {e}")


# ── SECTION 5: CONFIG VALUES ──────────────────────────────────
print("\n── CONFIG VALUES ───────────────────────────────────────")
try:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))

    required_blocks = ["detection", "tracking", "reid", "action", "anomaly", "fusion", "mlflow"]
    for block in required_blocks:
        if block in cfg:
            ok(f"config [{block}] block present")
        else:
            fail(f"config [{block}] block MISSING")

    mlflow_uri = cfg.get("mlflow", {}).get("tracking_uri", "")
    if mlflow_uri == "http://localhost:5001":
        ok(f"MLflow URI: {mlflow_uri}")
    else:
        fail(f"Wrong MLflow URI: '{mlflow_uri}' — must be http://localhost:5001")

    reid_model_name = cfg.get("reid", {}).get("model_name", "")
    if "osnet_ain" in reid_model_name:
        ok(f"Re-ID model name: {reid_model_name}")
    else:
        fail(f"Re-ID model name wrong: '{reid_model_name}' — must be osnet_ain variant")

    trk = cfg.get("tracking", {})
    if "track_thresh" in trk and "match_thresh" in trk:
        ok("ByteTrack config keys present (track_thresh, match_thresh)")
    else:
        fail("ByteTrack config keys missing — check config.yaml tracking block")
    for deepsort_key in ["n_init", "nn_budget", "embedder", "embedder_gpu"]:
        if deepsort_key in trk:
            fail(f"DeepSORT key '{deepsort_key}' still in tracking block — remove it")

except Exception as e:
    fail(f"config.yaml parse error  — {e}")


# ── SECTION 6: EPOCH SETTINGS ─────────────────────────────────
print("\n── EPOCH SETTINGS ──────────────────────────────────────")
epoch_checks = {
    "scripts/train_detector.py": "epochs=",
    "scripts/train_reid.py":     "MAX_EPOCH",
    "scripts/train_action.py":   "EPOCHS",
    "scripts/train_anomaly.py":  "EPOCHS",
    "scripts/train_fusion.py":   "EPOCHS",
}
for fpath, keyword in epoch_checks.items():
    try:
        src   = open(ROOT / fpath).read()
        lines = [l.strip() for l in src.splitlines() if keyword in l and "=" in l]
        if lines:
            ok(f"{fpath.split('/')[-1]}  —  {lines[0]}")
        else:
            fail(f"{fpath.split('/')[-1]}  — '{keyword}' setting not found")
    except Exception as e:
        fail(f"{fpath.split('/')[-1]}  — {e}")

# Warn if FREEZE_EPOCHS looks wrong for the set EPOCHS
try:
    src = open(ROOT / "scripts/train_action.py").read()
    import re
    epochs_match     = re.search(r"EPOCHS\s*=\s*(\d+)",       src)
    freeze_match     = re.search(r"FREEZE_EPOCHS\s*=\s*(\d+)", src)
    if epochs_match and freeze_match:
        epochs = int(epochs_match.group(1))
        freeze = int(freeze_match.group(1))
        if freeze >= epochs:
            fail(f"train_action.py — FREEZE_EPOCHS ({freeze}) >= EPOCHS ({epochs}). "
                 f"Backbone will never unfreeze. Set FREEZE_EPOCHS to ~20% of EPOCHS.")
        else:
            ok(f"train_action.py — EPOCHS={epochs}, FREEZE_EPOCHS={freeze} (OK)")
except Exception as e:
    fail(f"FREEZE_EPOCHS check  — {e}")


# ── SECTION 7: CRITICAL CODE CONSISTENCY ──────────────────────
print("\n── CRITICAL CONSISTENCY ────────────────────────────────")

# 1. output_proj key must match between train_anomaly.py and anomaly_detector.py
try:
    src_train = open(ROOT / "scripts/train_anomaly.py").read()
    src_infer = open(ROOT / "argus/anomaly/anomaly_detector.py").read()
    if "output_proj" in src_train and "output_proj" in src_infer:
        ok("output_proj key matches: train_anomaly.py ↔ anomaly_detector.py")
    else:
        fail("CRITICAL: output_proj MISMATCH between train_anomaly.py and "
             "anomaly_detector.py — trained model will fail to load")
except Exception as e:
    fail(f"output_proj check  — {e}")

# 2. np.asfarray shim in evaluate_mot17.py
try:
    lines = open(ROOT / "scripts/evaluate_mot17.py").readlines()
    numpy_idx = next((i for i, l in enumerate(lines) if "import numpy as np" in l), None)
    shim_idx  = next((i for i, l in enumerate(lines) if "np.asfarray" in l and "lambda" in l), None)
    if numpy_idx is not None and shim_idx is not None and shim_idx <= numpy_idx + 3:
        ok("np.asfarray shim present and correctly positioned in evaluate_mot17.py")
    else:
        fail("np.asfarray shim missing or incorrectly positioned in evaluate_mot17.py")
except Exception as e:
    fail(f"asfarray shim check  — {e}")

# 3. Tracker has no embedder param
try:
    src = open(ROOT / "argus/tracking/tracker.py").read()
    if "embedder" not in src and "DeepSort" not in src:
        ok("tracker.py — ByteTrack, no embedder param")
    else:
        fail("tracker.py still contains embedder/DeepSort references")
except Exception as e:
    fail(f"tracker.py check  — {e}")

# 4. OSNet-AIN in reid.py
try:
    src = open(ROOT / "argus/reid/reid.py").read()
    if "osnet_ain" in src:
        ok("reid.py — OSNet-AIN model name present")
    else:
        fail("reid.py — osnet_ain not found, may still use osnet_x1_0")
except Exception as e:
    fail(f"reid.py check  — {e}")


# ── SUMMARY ───────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  RESULT:  {PASS_COUNT} passed   {FAIL_COUNT} failed")
print(f"{'═'*55}")

if FAIL_COUNT > 0:
    print(f"  ❌  Fix all {FAIL_COUNT} failure(s) before running run_all.py\n")
    sys.exit(1)
else:
    print("  ✅  All checks passed — safe to start training")
    print("  Next command:  python run_all.py\n")
