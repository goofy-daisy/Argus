#!/usr/bin/env python3
"""
Argus Phase 7 — LSTM Autoencoder Anomaly Detection Training.

DO NOT RUN from Claude Code. Trigger manually when ready:
    python scripts/train_anomaly.py

Prerequisites:
    python scripts/prepare_mot17_trajectories.py  (already run in Phase 7 setup)

Training time: approximately 15-30 minutes on Apple Silicon MPS.
Output: argus/models/lstm_autoencoder.pth
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR    = PROJECT_ROOT / "argus" / "data" / "trajectories"
OUT_PATH    = PROJECT_ROOT / "argus" / "models" / "lstm_autoencoder.pth"

SEQ_LEN    = 30
FEAT_DIM   = 5
HIDDEN     = 64
N_LAYERS   = 2
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        dropout = 0.1 if N_LAYERS > 1 else 0.0
        self.encoder = nn.LSTM(
            FEAT_DIM, HIDDEN, N_LAYERS, batch_first=True, dropout=dropout
        )
        self.decoder = nn.LSTM(
            HIDDEN, HIDDEN, N_LAYERS, batch_first=True, dropout=dropout
        )
        self.output_proj = nn.Linear(HIDDEN, FEAT_DIM)

    def forward(self, x):
        batch, seq, _ = x.shape
        _, (h, c) = self.encoder(x)
        ctx = h[-1].unsqueeze(1).expand(batch, seq, HIDDEN)
        out, _ = self.decoder(ctx.contiguous(), (h, c))
        return self.output_proj(out)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            total += criterion(recon, batch).item()
    return total / len(loader)


def compute_per_sample_mse(model, loader, device):
    """Return MSE per sample as numpy array."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            scores.extend(mse.cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    tracking_uri = config["mlflow"]["tracking_uri"]

    # Load prepared sequences
    train_raw = np.load(str(DATA_DIR / "train_sequences.npy"))
    val_raw   = np.load(str(DATA_DIR / "val_sequences.npy"))
    logger.info("Train: %d  Val: %d  Shape: %s", len(train_raw), len(val_raw), train_raw.shape)

    # Compute and store standardisation stats from training set
    flat = train_raw.reshape(-1, FEAT_DIM)
    feat_mean = flat.mean(axis=0).astype(np.float32)
    feat_std  = flat.std(axis=0).astype(np.float32)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)   # avoid division by zero
    logger.info("Feature mean: %s", feat_mean.round(4))
    logger.info("Feature std:  %s", feat_std.round(4))

    # Standardise
    train_norm = ((train_raw - feat_mean) / feat_std).astype(np.float32)
    val_norm   = ((val_raw   - feat_mean) / feat_std).astype(np.float32)

    train_tensor = torch.FloatTensor(train_norm)
    val_tensor   = torch.FloatTensor(val_norm)

    train_loader = torch.utils.data.DataLoader(
        train_tensor, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_tensor, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False
    )

    model     = LSTMAutoencoder().to(torch.device(DEVICE))
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    EARLY_STOP_PATIENCE = 10
    MIN_LR              = 1e-5   # must match min_lr in ReduceLROnPlateau above
    best_val_loss       = float("inf")
    best_state          = None
    epochs_at_min_lr    = 0

    # MLflow
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        mlflow.start_run(tags={"pipeline_stage": "anomaly_training"})
        mlflow_active = True
    except Exception:
        pass

    try:
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, torch.device(DEVICE))
            vl_loss = eval_epoch(model, val_loader, criterion, torch.device(DEVICE))
            scheduler.step(vl_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.2e",
                epoch, EPOCHS, tr_loss, vl_loss, current_lr,
            )
            if mlflow_active:
                import mlflow
                mlflow.log_metrics({"train_loss": tr_loss, "val_loss": vl_loss}, step=epoch)

            if vl_loss < best_val_loss:
                best_val_loss    = vl_loss
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_at_min_lr = 0
                logger.info("  ↑ New best val_loss=%.6f", best_val_loss)
            else:
                # Track consecutive non-improving epochs when LR is at floor
                if current_lr <= MIN_LR:
                    epochs_at_min_lr += 1
                else:
                    epochs_at_min_lr = 0

            if epochs_at_min_lr >= EARLY_STOP_PATIENCE:
                logger.info(
                    "Early stopping at epoch %d: LR at min_lr=%.1e "
                    "for %d consecutive non-improving epochs.",
                    epoch, MIN_LR, EARLY_STOP_PATIENCE,
                )
                if mlflow_active:
                    import mlflow
                    mlflow.log_param("early_stop_epoch", epoch)
                break
    finally:
        if mlflow_active:
            import mlflow
            mlflow.end_run()

    # Load best weights and calibrate threshold
    model.load_state_dict(best_state)
    model.eval()

    val_scores = compute_per_sample_mse(model, val_loader, torch.device(DEVICE))
    threshold  = float(val_scores.mean() + 3.0 * val_scores.std())
    logger.info(
        "Threshold calibration — mean=%.6f  std=%.6f  threshold(mean+3σ)=%.6f",
        val_scores.mean(), val_scores.std(), threshold,
    )

    # Save checkpoint with everything needed for inference
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":    model.state_dict(),
            "threshold":     threshold,
            "feat_mean":     feat_mean.tolist(),
            "feat_std":      feat_std.tolist(),
            "val_mean":      float(val_scores.mean()),
            "val_std":       float(val_scores.std()),
            "best_val_loss": best_val_loss,
        },
        str(OUT_PATH),
    )
    logger.info("Checkpoint saved to %s", OUT_PATH)
    logger.info("Next step: python scripts/evaluate_anomaly.py")


if __name__ == "__main__":
    main()
