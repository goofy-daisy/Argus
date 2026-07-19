#!/usr/bin/env python3
"""
Argus Phase 5 — OSNet-x1.0 Re-ID Training on Market-1501.

DO NOT RUN this script from Claude Code. Trigger manually when ready:
    python scripts/train_reid.py

Training time: approximately 45–90 minutes on Apple Silicon MPS.
Checkpoints saved to: runs/reid/
Final weights:        argus/models/osnet_ain_x1_0_market1501.pth

NOTE: torchreid's built-in engine is not MPS-aware — it moves inputs
to CPU while the model is on MPS, crashing on the first training step.
This script uses a custom PyTorch training loop instead.
torchreid is still used for model construction (build_model) and weight
loading utilities only.

After training completes:
    python scripts/export_reid_weights.py
    python scripts/evaluate_reid.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH      = PROJECT_ROOT / "config.yaml"
MARKET_TRAIN_DIR = (
    PROJECT_ROOT
    / "argus" / "data" / "Market-1501"
    / "Market-1501-v15.09.15" / "bounding_box_train"
)
SAVE_DIR = PROJECT_ROOT / "runs" / "reid"
OUT_PATH = PROJECT_ROOT / "argus" / "models" / "osnet_ain_x1_0_market1501.pth"

MAX_EPOCH     = 60
WARMUP_EPOCHS = 5
BATCH_SIZE    = 64
LR            = 0.0035
WEIGHT_DECAY  = 5e-4
MILESTONES    = [30, 50]
NUM_WORKERS   = 0        # MPS requires 0
DEVICE        = "mps"


# ── Dataset ───────────────────────────────────────────────────────────────────

class Market1501Dataset(torch.utils.data.Dataset):
    """Market-1501 bounding_box_train directory for softmax ReID pre-training.

    Parses person ID from the first 4 characters of each filename, skips
    junk/background images (pid <= 0), and maps sorted unique PIDs to
    sequential class indices 0..N-1.
    """

    def __init__(self, data_dir: Path, transform: transforms.Compose) -> None:
        self.transform = transform

        raw: list = []
        all_pids: set = set()
        for img_path in sorted(data_dir.glob("*.jpg")):
            try:
                pid = int(img_path.stem[:4])
            except ValueError:
                continue
            if pid <= 0:
                continue
            all_pids.add(pid)
            raw.append((img_path, pid))

        # Sequential label mapping: sorted pid → 0-indexed class label
        self.pid_to_label = {pid: idx for idx, pid in enumerate(sorted(all_pids))}
        self.num_classes   = len(self.pid_to_label)
        self.samples       = [(p, self.pid_to_label[pid]) for p, pid in raw]

        logger.info(
            "Market-1501 train: %d images, %d identities",
            len(self.samples), self.num_classes,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img    = Image.open(str(img_path)).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


# ── Transforms ────────────────────────────────────────────────────────────────

def build_transforms() -> transforms.Compose:
    """Training transforms for Market-1501 person crops.

    RandomErasing (occlusion simulation) is included — the single
    highest-impact augmentation for ReID rank-1 accuracy.
    """
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    ])


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total      = 0
    for imgs, pids in loader:
        imgs = imgs.to(torch.device(device))
        pids = pids.to(torch.device(device))
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, pids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / max(total, 1)


def log_to_mlflow(metrics: dict, params: dict, tracking_uri: str) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        with mlflow.start_run(
            tags={"pipeline_stage": "reid_training", "dataset": "market1501"}
        ):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
        logger.info("Metrics logged to MLflow.")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not MARKET_TRAIN_DIR.exists():
        logger.error(
            "Market-1501 training directory not found at %s", MARKET_TRAIN_DIR
        )
        sys.exit(1)

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    tracking_uri = config["mlflow"]["tracking_uri"]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Device        : %s", DEVICE)
    logger.info("LR            : %.4f  (linear warmup for %d epochs)", LR, WARMUP_EPOCHS)
    logger.info("Max epochs    : %d", MAX_EPOCH)
    logger.info("LR milestones : %s  (gamma=0.1)", MILESTONES)
    logger.info("Batch size    : %d", BATCH_SIZE)

    # ── Dataset and DataLoader ─────────────────────────────────────────────────
    dataset = Market1501Dataset(MARKET_TRAIN_DIR, build_transforms())
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    import torchreid

    model = torchreid.models.build_model(
        name="osnet_ain_x1_0",
        num_classes=dataset.num_classes,
        loss="softmax",
        pretrained=True,   # ImageNet backbone as starting point
        use_gpu=False,     # device managed manually below
    )
    model = model.to(torch.device(DEVICE))
    logger.info(
        "OSNet-AIN-x1.0 built: num_classes=%d  device=%s", dataset.num_classes, DEVICE
    )

    # ── Loss, optimiser, and scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        amsgrad=True,
    )
    # MultiStepLR governs post-warmup epochs; milestones are epoch indices (0-based)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=0.1
    )

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_active = False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        mlflow.start_run(
            tags={"pipeline_stage": "reid_training", "dataset": "market1501"}
        )
        mlflow_active = True
    except Exception:
        pass

    avg_loss = 0.0   # captured outside loop for final logging

    try:
        for epoch in range(MAX_EPOCH):
            # ── Linear LR warmup for first WARMUP_EPOCHS epochs ───────────────
            if epoch < WARMUP_EPOCHS:
                warmup_lr = LR * (epoch + 1) / WARMUP_EPOCHS
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            avg_loss   = train_one_epoch(model, loader, criterion, optimizer, DEVICE)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %3d/%d  loss=%.4f  lr=%.2e",
                epoch + 1, MAX_EPOCH, avg_loss, current_lr,
            )

            # Step the MultiStepLR only after the warmup phase
            if epoch >= WARMUP_EPOCHS:
                scheduler.step()

            if mlflow_active:
                import mlflow

                mlflow.log_metrics(
                    {"train_loss": avg_loss, "lr": current_lr},
                    step=epoch + 1,
                )

            # Periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt_path = SAVE_DIR / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(
                    {
                        "state_dict":  model.state_dict(),
                        "epoch":       epoch + 1,
                        "train_loss":  avg_loss,
                        "num_classes": dataset.num_classes,
                    },
                    str(ckpt_path),
                )
                logger.info("Checkpoint saved → %s", ckpt_path)

    finally:
        if mlflow_active:
            import mlflow

            mlflow.end_run()

    # ── Save final weights in format compatible with export_reid_weights.py ────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict":  model.state_dict(),
            "epoch":       MAX_EPOCH,
            "train_loss":  avg_loss,
            "num_classes": dataset.num_classes,
        },
        str(OUT_PATH),
    )
    logger.info("Final weights saved to %s", OUT_PATH)

    params = {
        "model":          "osnet_ain_x1_0",
        "dataset":        "market1501",
        "lr":             LR,
        "max_epoch":      MAX_EPOCH,
        "warmup_epochs":  WARMUP_EPOCHS,
        "milestones":     str(MILESTONES),
        "batch_size":     BATCH_SIZE,
        "weight_decay":   WEIGHT_DECAY,
        "random_erasing": True,
    }
    log_to_mlflow({"final_train_loss": avg_loss}, params, tracking_uri)

    logger.info("Training complete.")
    logger.info("Next step: python scripts/export_reid_weights.py")


if __name__ == "__main__":
    main()
