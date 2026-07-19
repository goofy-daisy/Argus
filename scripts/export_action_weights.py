#!/usr/bin/env python3
"""
Argus Phase 6 — Export trained X3D-S weights.

Run once after train_action.py completes:
    python scripts/export_action_weights.py

Reads:  argus/models/action_training/best_model.pth
Writes: argus/models/x3d_s_argus.pth
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC = PROJECT_ROOT / "argus" / "models" / "action_training" / "best_model.pth"
DST = PROJECT_ROOT / "argus" / "models" / "x3d_s_argus.pth"


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: Checkpoint not found at {SRC}")
        print("Run scripts/train_action.py first.")
        sys.exit(1)

    shutil.copy(str(SRC), str(DST))
    print(f"Weights exported to {DST}")
    print("Next step: python scripts/evaluate_action.py")


if __name__ == "__main__":
    main()
