#!/usr/bin/env python3
"""
Argus Phase 7 — MOT17 Trajectory Data Preparation.

Extracts 5-dimensional trajectory feature sequences from all 7 MOT17
FRCNN training sequences and saves them as .npy arrays.

Run from project root with venv active:
    python scripts/prepare_mot17_trajectories.py

Outputs:
    argus/data/trajectories/train_sequences.npy  (N_train, 30, 5)
    argus/data/trajectories/val_sequences.npy    (N_val, 30, 5)
    argus/data/trajectories/metadata.json
"""
from __future__ import annotations

import configparser
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MOT17_ROOT = PROJECT_ROOT / "argus" / "data" / "MOT17" / "train"
OUT_DIR    = PROJECT_ROOT / "argus" / "data" / "trajectories"
SEQ_LEN    = 30
STRIDE     = 5
VAL_RATIO  = 0.20
SEED       = 42

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def read_seqinfo(seq_dir: Path):
    parser = configparser.ConfigParser()
    parser.read(str(seq_dir / "seqinfo.ini"))
    return {
        "width":  int(parser["Sequence"]["imWidth"]),
        "height": int(parser["Sequence"]["imHeight"]),
        "length": int(parser["Sequence"]["seqLength"]),
    }


def extract_sequences(gt_path: Path, frame_width: int, frame_height: int):
    """Extract sliding-window trajectory sequences from one gt.txt file."""
    gt = pd.read_csv(
        str(gt_path),
        header=None,
        names=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"],
    )
    gt = gt[(gt["conf"] == 1) & (gt["class"] == 1)].copy()

    sequences = []
    for track_id, track in gt.groupby("id"):
        track = track.sort_values("frame").reset_index(drop=True)
        if len(track) < SEQ_LEN:
            continue

        # Compute feature vectors
        cx = (track["x"] + track["w"] / 2.0) / frame_width
        cy = (track["y"] + track["h"] / 2.0) / frame_height
        ar = track["w"] / (track["h"] + 1e-6)

        # bfill() propagates the second frame's velocity back to frame 0,
        # avoiding the all-zero first-frame bias that fillna(0.0) introduced.
        vx = cx.diff().bfill()
        vy = cy.diff().bfill()

        feats = np.stack(
            [cx.values, cy.values, vx.values, vy.values, ar.values],
            axis=1,
        ).astype(np.float32)

        # Sliding window
        for start in range(0, len(feats) - SEQ_LEN + 1, STRIDE):
            window = feats[start : start + SEQ_LEN]
            sequences.append(window)

    return sequences


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sequences = []
    metadata = {"sequences": {}, "total": 0, "train": 0, "val": 0}

    for seq_name in SEQUENCES:
        seq_dir = MOT17_ROOT / seq_name
        gt_path = seq_dir / "gt" / "gt.txt"
        info    = read_seqinfo(seq_dir)
        seqs    = extract_sequences(gt_path, info["width"], info["height"])
        all_sequences.extend(seqs)
        metadata["sequences"][seq_name] = {
            "windows": len(seqs),
            "width":   info["width"],
            "height":  info["height"],
        }
        logger.info("%-22s  frames=%d  windows=%d", seq_name, info["length"], len(seqs))

    arr = np.array(all_sequences, dtype=np.float32)   # (N, 30, 5)
    logger.info("Total windows: %d  shape: %s", len(arr), arr.shape)

    # Shuffle and split
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(arr))
    arr = arr[idx]

    n_val   = max(1, int(len(arr) * VAL_RATIO))
    n_train = len(arr) - n_val
    train   = arr[:n_train]
    val     = arr[n_train:]

    np.save(str(OUT_DIR / "train_sequences.npy"), train)
    np.save(str(OUT_DIR / "val_sequences.npy"),   val)

    metadata["total"]    = int(len(arr))
    metadata["train"]    = int(n_train)
    metadata["val"]      = int(n_val)
    metadata["seq_len"]  = SEQ_LEN
    metadata["feat_dim"] = 5
    metadata["features"] = ["x_norm", "y_norm", "vx", "vy", "aspect_ratio"]

    with open(OUT_DIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Train: %d  Val: %d", n_train, n_val)
    logger.info("Saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
