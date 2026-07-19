#!/usr/bin/env python3
"""
Argus Phase 5 — Export trained OSNet weights from torchreid checkpoint.

Run once after train_reid.py completes:
    python scripts/export_reid_weights.py

Reads:  argus/models/reid_training/model.pth.tar-best
Writes: argus/models/osnet_x1_0_market1501.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_PATH = PROJECT_ROOT / "argus" / "models" / "reid_training" / "model.pth.tar-best"
OUT_PATH = PROJECT_ROOT / "argus" / "models" / "osnet_x1_0_market1501.pth"


def main() -> None:
    if not CKPT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        print("Run scripts/train_reid.py first.")
        sys.exit(1)

    checkpoint = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    torch.save(state_dict, str(OUT_PATH))
    print(f"Weights exported to {OUT_PATH}")

    if "rank1" in checkpoint:
        print(f"Training Rank-1 CMC (from checkpoint): {checkpoint['rank1']:.4f}")


if __name__ == "__main__":
    main()
