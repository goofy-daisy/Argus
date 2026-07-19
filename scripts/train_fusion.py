#!/usr/bin/env python3
"""
Argus Phase 8 — Attention Gate Fusion Training on FLIR ADAS.

DO NOT RUN from Claude Code. Trigger manually when ready:
    python scripts/train_fusion.py

Training time: approximately 30-60 minutes on Apple Silicon MPS.
Output: argus/models/attention_fusion.pth
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from argus.fusion.fusion import _build_fusion_net

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUT_PATH    = PROJECT_ROOT / "argus" / "models" / "attention_fusion.pth"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE    = 8
EPOCHS        = 30
LR            = 1e-4
IMG_SIZE      = 320     # resize for memory efficiency during training
LAMBDA_GATE   = 0.1     # gate entropy regularisation weight
WARMUP_EPOCHS = 5      # linear LR warmup before cosine schedule


RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FLIRPairedDataset(torch.utils.data.Dataset):
    """Paired FLIR ADAS RGB + thermal images for self-supervised fusion training."""

    def __init__(self, rgb_dir: Path, thm_dir: Path, img_size: int = 320) -> None:
        self.img_size = img_size
        rgb_files = sorted((rgb_dir / "data").glob("*.jpg"))
        thm_files = sorted(
            list((thm_dir / "data").glob("*.jpg")) +
            list((thm_dir / "data").glob("*.jpeg"))
        )
        n = min(len(rgb_files), len(thm_files))
        self.pairs = list(zip(rgb_files[:n], thm_files[:n]))
        logger.info("Paired samples: %d (sequential pairing, rgb=%d thermal=%d)",
                    len(self.pairs), len(rgb_files), len(thm_files))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rgb_path, thm_path = self.pairs[idx]
        rgb = cv2.imread(str(rgb_path))
        thm = cv2.imread(str(thm_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or thm is None:
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            thm = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        rgb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        thm = cv2.resize(thm, (self.img_size, self.img_size))
        rgb_f = (rgb.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
        thm_f = thm.astype(np.float32) / 255.0
        rgb_t = torch.FloatTensor(rgb_f.transpose(2, 0, 1))
        thm_t = torch.FloatTensor(thm_f[np.newaxis])
        return rgb_t, thm_t


def main() -> None:
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    cfg = config["fusion"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    rgb_train = PROJECT_ROOT / cfg["flir_rgb_train"]
    thm_train = PROJECT_ROOT / cfg["flir_thermal_train"]

    dataset = FLIRPairedDataset(rgb_train, thm_train, IMG_SIZE)
    if len(dataset) == 0:
        logger.error("No paired FLIR samples found. Check dataset paths.")
        sys.exit(1)

    # ── 80/20 train/val split (fixed seed for reproducibility) ────────────────
    n_val   = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Train: %d  Val: %d", n_train, n_val)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = _build_fusion_net(feature_channels=cfg["feature_channels"])
    model = model.to(torch.device(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # Cosine schedule covers only the post-warmup epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS)
    )
    criterion = nn.L1Loss()

    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        mlflow.start_run(tags={"pipeline_stage": "fusion_training"})
        mlflow_active = True
    except Exception:
        pass

    best_val_loss = float("inf")
    best_state    = None

    try:
        for epoch in range(1, EPOCHS + 1):
            # ── Linear LR warmup ──────────────────────────────────────────────
            if epoch <= WARMUP_EPOCHS:
                warmup_lr = LR * epoch / WARMUP_EPOCHS
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # ── Training ──────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0
            for rgb_t, thm_t in train_loader:
                rgb_t = rgb_t.to(DEVICE)
                thm_t = thm_t.to(DEVICE)
                optimizer.zero_grad()
                fused, gate = model(rgb_t, thm_t)
                recon_loss = criterion(fused, rgb_t)   # reconstruction target: RGB
                # Gate entropy regularisation: prevent gate from collapsing to
                # all-0 (ignore thermal) or all-1 (ignore RGB). Maximise entropy
                # by subtracting the negative entropy term from the total loss.
                gate_entropy = -(
                    gate * torch.log(gate + 1e-8)
                    + (1 - gate) * torch.log(1 - gate + 1e-8)
                ).mean()
                total_loss_batch = recon_loss - LAMBDA_GATE * gate_entropy
                total_loss_batch.backward()
                optimizer.step()
                total_loss += recon_loss.item()
            avg_train = total_loss / len(train_loader)

            # ── Validation ────────────────────────────────────────────────────
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for rgb_t, thm_t in val_loader:
                    rgb_t = rgb_t.to(DEVICE)
                    thm_t = thm_t.to(DEVICE)
                    fused, _ = model(rgb_t, thm_t)
                    val_total += criterion(fused, rgb_t).item()
            avg_val = val_total / len(val_loader)

            # Step cosine scheduler only after warmup phase
            if epoch > WARMUP_EPOCHS:
                scheduler.step()

            logger.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.2e",
                epoch, EPOCHS, avg_train, avg_val,
                optimizer.param_groups[0]["lr"],
            )
            if mlflow_active:
                import mlflow
                mlflow.log_metrics(
                    {"train_loss": avg_train, "val_loss": avg_val}, step=epoch
                )

            # Checkpoint selection on val loss (not train loss)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                logger.info("  ↑ New best val_loss=%.6f", best_val_loss)
    finally:
        if mlflow_active:
            import mlflow
            mlflow.end_run()

    model.load_state_dict(best_state)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(OUT_PATH))
    logger.info("Fusion model saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
