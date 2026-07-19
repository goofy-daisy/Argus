#!/usr/bin/env python3
"""
Argus — Full Training and Evaluation Pipeline
Runs all 5 training scripts in sequence, exports weights, then runs all 5 evaluation scripts.
All output is printed to the terminal in real time AND written to run_all.log in the project root.

Usage (from project root with venv active):
    python run_all.py

DO NOT RUN from Claude Code. User-triggered only.
If this process is interrupted, restart from the beginning — do not run partial sequences.
"""
import os
import re
import sys
import subprocess
import time
import yaml
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
PYTHON = sys.executable

# Force MLflow file store for all subprocesses
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

LOG_PATH = ROOT / "run_all.log"
_log_fh  = open(LOG_PATH, "w", buffering=1)

# ── LOGGING ───────────────────────────────────────────────────

def log(msg: str = ""):
    print(msg, flush=True)
    _log_fh.write(msg + "\n")
    _log_fh.flush()

def log_header(title: str):
    log()
    log("═" * 62)
    log(f"  {title}")
    log("═" * 62)

def log_step(title: str):
    log()
    log("─" * 62)
    log(f"  ▶  {title}")
    log("─" * 62)


# ── STEP RUNNER ───────────────────────────────────────────────

def run_step(label: str, script: str, check_path: str | None = None) -> None:
    """Run a training or evaluation script as a subprocess.

    Output is streamed directly to the terminal (not captured) so progress
    is visible in real time. Step results are logged to run_all.log.
    Aborts the entire pipeline on non-zero exit or missing output file.
    """
    if check_path and (ROOT / check_path).exists():
        mb = (ROOT / check_path).stat().st_size / 1e6
        log(f"  ⏭  SKIP (weight exists): {label} — {check_path} ({mb:.1f} MB)")
        return
    log_step(label)
    t0 = time.perf_counter()

    result = subprocess.run(
        [PYTHON, str(ROOT / script)],
        cwd=str(ROOT),
        # stdout/stderr not captured — streams directly to terminal
    )

    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log(f"\n  ✗  FAILED: {label}  (exit code {result.returncode})")
        log(f"     Elapsed: {elapsed / 60:.1f} min")
        log(f"     Pipeline aborted. Fix the error and restart run_all.py from scratch.")
        _log_fh.close()
        sys.exit(1)

    if check_path:
        p = ROOT / check_path
        if not p.exists():
            log(f"\n  ✗  FAILED: {label}")
            log(f"     Expected output file not found: {check_path}")
            log(f"     Pipeline aborted.")
            _log_fh.close()
            sys.exit(1)
        mb = p.stat().st_size / 1e6
        log(f"\n  ✅  {label}")
        log(f"     Output: {p.name}  ({mb:.1f} MB)  |  Time: {elapsed / 60:.1f} min")
    else:
        log(f"\n  ✅  {label}  |  Time: {elapsed / 60:.1f} min")


# ── CONFIG HELPERS ────────────────────────────────────────────

def update_config_detection_model(model_path_relative: str) -> None:
    """Update config.yaml detection.model_path after training."""
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cfg["detection"]["model_path"] = model_path_relative
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    log(f"  config.yaml → detection.model_path: {model_path_relative}")


def patch_evaluate_mot17(use_finetuned: bool) -> None:
    """Patch evaluate_mot17.py MODEL_PATH to use fine-tuned or base weights.

    evaluate_mot17.py hardcodes MODEL_PATH to yolov8m.pt. This function
    patches it before evaluation and restores it after.
    """
    script_path = ROOT / "scripts" / "evaluate_mot17.py"
    src = script_path.read_text()

    if use_finetuned:
        patched = re.sub(
            r'(MODEL_PATH\s*=\s*str\(\s*PROJECT_ROOT\s*/[^)]*?)"yolov8m\.pt"',
            r'\1"yolov8m_finetuned.pt"',
            src,
        )
        if "yolov8m_finetuned.pt" in patched:
            script_path.write_text(patched)
            log("  evaluate_mot17.py → patched to use yolov8m_finetuned.pt")
        else:
            log("  WARNING: evaluate_mot17.py patch did not apply — "
                "evaluation will use base COCO weights instead of fine-tuned.")
    else:
        patched = re.sub(
            r'(MODEL_PATH\s*=\s*str\(\s*PROJECT_ROOT\s*/[^)]*?)"yolov8m_finetuned\.pt"',
            r'\1"yolov8m.pt"',
            src,
        )
        script_path.write_text(patched)
        log("  evaluate_mot17.py → restored to yolov8m.pt")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.perf_counter()

    log_header("ARGUS — FULL TRAINING + EVALUATION PIPELINE")
    log(f"  Log file : {LOG_PATH}")
    log(f"  Python   : {PYTHON}")
    log(f"  MLFLOW_ALLOW_FILE_STORE = true")
    log(f"  Started  : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── PHASE 7: ANOMALY ──────────────────────────────────────
    run_step(
        label      = "Phase 7 — LSTM Anomaly Training",
        script     = "scripts/train_anomaly.py",
        check_path = "argus/models/lstm_autoencoder.pth",
    )

    # ── PHASE 8: FUSION ───────────────────────────────────────
    run_step(
        label      = "Phase 8 — Attention Fusion Training",
        script     = "scripts/train_fusion.py",
        check_path = "argus/models/attention_fusion.pth",
    )

    # ── PHASE 5: RE-ID ────────────────────────────────────────
    run_step(
        label      = "Phase 5 — OSNet-AIN Re-ID Training",
        script     = "scripts/train_reid.py",
        check_path = "argus/models/osnet_ain_x1_0_market1501.pth",
    )
    run_step(
        label      = "Phase 5 — Re-ID Weight Export",
        script     = "scripts/export_reid_weights.py",
        check_path = "argus/models/osnet_ain_x1_0_market1501.pth",
    )

    # ── PHASE 4: DETECTION ────────────────────────────────────
    run_step(
        label      = "Phase 4 — YOLOv8m Detection Fine-Tuning",
        script     = "scripts/train_detector.py",
        check_path = "argus/models/yolov8m_finetuned.pt",
    )

    # ── PHASE 6: ACTION ───────────────────────────────────────
    run_step(
        label      = "Phase 6 — X3D-S Action Training",
        script     = "scripts/train_action.py",
    )
    run_step(
        label      = "Phase 6 — Action Weight Export",
        script     = "scripts/export_action_weights.py",
        check_path = "argus/models/x3d_s_argus.pth",
    )

    # ── EVALUATION SETUP ──────────────────────────────────────
    log()
    log("─" * 62)
    log("  Configuring evaluation to use fine-tuned models")
    log("─" * 62)
    update_config_detection_model("argus/models/yolov8m_finetuned.pt")
    patch_evaluate_mot17(use_finetuned=True)

    # ── PHASE 4: DETECTION EVAL ───────────────────────────────
    run_step(
        label  = "Phase 4 — Detection Evaluation  (MOT17-02-FRCNN)",
        script = "scripts/evaluate_mot17.py",
    )

    # ── PHASE 5: RE-ID EVAL ───────────────────────────────────
    run_step(
        label  = "Phase 5 — Re-ID Evaluation       (Market-1501)",
        script = "scripts/evaluate_reid.py",
    )

    # ── PHASE 6: ACTION EVAL ──────────────────────────────────
    run_step(
        label  = "Phase 6 — Action Evaluation      (UCF101 proxy)",
        script = "scripts/evaluate_action.py",
    )

    # ── PHASE 7: ANOMALY EVAL ─────────────────────────────────
    run_step(
        label  = "Phase 7 — Anomaly Evaluation",
        script = "scripts/evaluate_anomaly.py",
    )

    # ── PHASE 8: CALIBRATION EVAL ─────────────────────────────
    run_step(
        label  = "Phase 8 — Calibration + Fusion Evaluation  (FLIR ADAS)",
        script = "scripts/calibrate_and_evaluate.py",
    )

    # ── RESTORE ───────────────────────────────────────────────
    patch_evaluate_mot17(use_finetuned=False)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    hours, rem    = divmod(total_elapsed, 3600)
    minutes       = rem // 60

    log()
    log_header("ALL TRAINING AND EVALUATION COMPLETE")
    log(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total    : {int(hours)}h {int(minutes)}m")
    log(f"  Log file : {LOG_PATH}")
    log()

    _log_fh.close()
