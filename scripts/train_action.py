#!/usr/bin/env python3
"""
Argus Phase 6 — X3D-S Action Recognition Training on UCF101 Proxy Classes.

DO NOT RUN this script from Claude Code. Trigger manually when ready:
    python scripts/train_action.py

Training time: approximately 1–2 hours on Apple Silicon MPS.
Best checkpoint saved to: argus/models/action_training/best_model.pth
After training:
    python scripts/export_action_weights.py
    python scripts/evaluate_action.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.transforms import ColorJitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# UCF101 root — keep in sync with dataset section 4 of project docs
UCF101_ROOT = PROJECT_ROOT / "argus" / "data" / "UCF101"
SAVE_DIR    = PROJECT_ROOT / "argus" / "models" / "action_training"

NUM_FRAMES          = 16
CLIP_SIZE           = 182
BATCH_SIZE          = 8     # increased from 4 for stable BatchNorm statistics
EPOCHS              = 40
HEAD_LR             = 1e-4
WEIGHT_DECAY        = 1e-4
FREEZE_EPOCHS       = 8
EARLY_STOP_PATIENCE = 8     # stop if val_acc stagnates for 8 consecutive epochs
NUM_WORKERS         = 0     # MPS requires 0

ARGUS_LABELS = ["normal", "loitering", "running", "falling", "crowd_formation"]

PROXY_MAP = {
    "normal":           ["TaiChi", "WalkingWithDog"],
    "loitering":        ["Knitting", "ShavingBeard"],
    "running":          ["Basketball", "SoccerPenalty"],
    "falling":          ["FloorGymnastics", "LongJump"],
    "crowd_formation":  ["BandMarching", "MilitaryParade"],
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

class UCF101ProxyDataset(torch.utils.data.Dataset):
    """UCF101 video dataset filtered to Argus proxy classes.

    video_mean and video_std are passed from config so they stay in sync
    with ActionClassifier.extract_clip() at inference time.
    """

    def __init__(
        self,
        ucf101_root: Path,
        split: str,
        num_frames: int = 16,
        clip_size: int = 182,
        augment: bool = False,
        video_mean: list | None = None,
        video_std: list | None = None,
    ) -> None:
        self.ucf101_root = ucf101_root
        self.num_frames  = num_frames
        self.clip_size   = clip_size
        self.augment     = augment
        self.video_mean  = np.array(
            video_mean if video_mean else [0.45, 0.45, 0.45], dtype=np.float32
        )
        self.video_std   = np.array(
            video_std if video_std else [0.225, 0.225, 0.225], dtype=np.float32
        )
        # Per-frame color jitter applied during training augmentation
        self._color_jitter = ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

        # Build UCF101 class → argus label index mapping
        self.ucf_to_argus: dict = {}
        for argus_label, ucf_classes in PROXY_MAP.items():
            argus_idx = ARGUS_LABELS.index(argus_label)
            for ucf_cls in ucf_classes:
                self.ucf_to_argus[ucf_cls] = argus_idx

        split_file = (
            ucf101_root / "ucfTrainTestlist" /
            ("trainlist01.txt" if split == "train" else "testlist01.txt")
        )
        self.samples: list = []
        with open(split_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                video_rel  = parts[0]
                class_name = video_rel.split("/")[0]
                if class_name in self.ucf_to_argus:
                    video_path = ucf101_root / "UCF-101" / video_rel
                    if video_path.exists():
                        label = self.ucf_to_argus[class_name]
                        self.samples.append((str(video_path), label))

        logger.info(
            "UCF101 %s split: %d clips across %d proxy classes",
            split, len(self.samples), len(self.ucf_to_argus),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        clip   = self._preprocess(frames)
        return clip, label

    def _load_frames(self, video_path: str):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.num_frames

        if total <= self.num_frames:
            raw_indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        elif self.augment:
            start       = np.random.randint(0, total - self.num_frames + 1)
            raw_indices = list(range(start, start + self.num_frames))
        else:
            raw_indices = np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()

        frames = []
        for fi in raw_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(
                    np.zeros((self.clip_size, self.clip_size, 3), dtype=np.uint8)
                )
        cap.release()
        return frames[: self.num_frames]

    def _preprocess(self, frames) -> torch.Tensor:
        clip = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.augment:
                # Random horizontal flip
                if np.random.random() > 0.5:
                    rgb = cv2.flip(rgb, 1)
                # Widened scale range: 1.0–1.4 (was 1.0–1.15)
                scale = np.random.uniform(1.0, 1.4)
                nh    = int(self.clip_size * scale)
                nw    = int(self.clip_size * scale)
                rgb   = cv2.resize(rgb, (nw, nh))
                y0    = np.random.randint(0, nh - self.clip_size + 1)
                x0    = np.random.randint(0, nw - self.clip_size + 1)
                rgb   = rgb[y0 : y0 + self.clip_size, x0 : x0 + self.clip_size]
                # Per-frame color jitter applied after spatial crop
                pil_img = Image.fromarray(rgb)
                pil_img = self._color_jitter(pil_img)
                rgb     = np.array(pil_img)
            else:
                rgb = cv2.resize(rgb, (self.clip_size, self.clip_size))
            clip.append(rgb)

        # Stack and normalise using config statistics (in sync with inference)
        arr = np.stack(clip, axis=0).astype(np.float32) / 255.0
        arr = (arr - self.video_mean) / self.video_std
        arr = arr.transpose(3, 0, 1, 2)   # (C, T, H, W)
        return torch.FloatTensor(arr)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """Load X3D-S with Kinetics-400 pretrained backbone, replace classification head.

    # KEEP IN SYNC WITH ActionClassifier._replace_head() in
    # argus/action/action_classifier.py — both must iterate named_modules()
    # identically to find and replace the last nn.Linear.
    """
    logger.info("Downloading X3D-S Kinetics-400 pretrained backbone...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "x3d_s",
        pretrained=True,   # downloads Kinetics-400 weights (~100MB, once)
    )

    # Replace last Linear layer with num_classes output
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last_module = name, module

    if last_name is not None:
        in_feats = last_module.in_features
        parts    = last_name.split(".")
        parent   = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, parts[-1], nn.Linear(in_feats, num_classes, bias=True))
        logger.info("Head replaced: Linear(%d → %d)", in_feats, num_classes)

    # Replace any Sigmoid/Softmax activation with Identity
    for name, module in model.named_modules():
        if isinstance(module, (nn.Sigmoid, nn.Softmax)):
            p   = name.split(".")
            par = model
            for part in p[:-1]:
                par = par[int(part)] if part.isdigit() else getattr(par, part)
            setattr(par, p[-1], nn.Identity())

    return model


def get_head_param_names(model: nn.Module) -> set:
    """Return parameter names belonging to the last (replaced) Linear layer."""
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    if last_name is None:
        return set()
    return {f"{last_name}.weight", f"{last_name}.bias"}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(clips)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits        = model(clips)
            loss          = criterion(logits, labels)
            total_loss   += loss.item()
            correct      += (logits.argmax(1) == labels).sum().item()
            total        += labels.size(0)
    return total_loss / len(loader), correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    tracking_uri = config["mlflow"]["tracking_uri"]
    action_cfg   = config["action"]

    # Read normalisation stats from config — must match inference (ActionClassifier)
    video_mean = action_cfg.get("video_mean", [0.45, 0.45, 0.45])
    video_std  = action_cfg.get("video_std",  [0.225, 0.225, 0.225])

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Device : %s  |  Epochs : %d  |  Batch : %d  |  LR(head) : %.0e",
        DEVICE, EPOCHS, BATCH_SIZE, HEAD_LR,
    )
    logger.info(
        "Freeze : %d epochs → backbone=1e-5 + head=%.0e with CosineAnnealingLR",
        FREEZE_EPOCHS, HEAD_LR,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = UCF101ProxyDataset(
        UCF101_ROOT, "train", NUM_FRAMES, CLIP_SIZE,
        augment=True, video_mean=video_mean, video_std=video_std,
    )
    val_dataset = UCF101ProxyDataset(
        UCF101_ROOT, "test", NUM_FRAMES, CLIP_SIZE,
        augment=False, video_mean=video_mean, video_std=video_std,
    )

    # ── Class-balanced sampler: 1 / class_count per sample ────────────────────
    label_counts   = Counter(label for _, label in train_dataset.samples)
    sample_weights = [1.0 / label_counts[label] for _, label in train_dataset.samples]
    sampler        = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,       # mutually exclusive with shuffle=True
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=len(ARGUS_LABELS))
    model = model.to(torch.device(DEVICE))

    # ── Phase 1: freeze backbone, train head only ─────────────────────────────
    head_param_names = get_head_param_names(model)
    for name, param in model.named_parameters():
        if name not in head_param_names:
            param.requires_grad = False
    logger.info(
        "Backbone frozen; head has %d trainable params.",
        sum(p.numel() for n, p in model.named_parameters() if n in head_param_names),
    )

    head_params_list = [p for n, p in model.named_parameters() if n in head_param_names]
    optimizer        = torch.optim.AdamW(
        head_params_list, lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    criterion        = nn.CrossEntropyLoss()
    scheduler        = None   # created when backbone is unfrozen

    best_val_acc      = 0.0
    best_ckpt_path    = SAVE_DIR / "best_model.pth"
    epochs_no_improve = 0

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_active = False
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("argus")
        mlflow.start_run(tags={"pipeline_stage": "action_training"})
        mlflow_active = True
    except Exception:
        pass

    try:
        for epoch in range(1, EPOCHS + 1):

            # ── Phase 2: unfreeze backbone + differential LR at epoch 11 ──────
            if epoch == FREEZE_EPOCHS + 1:
                logger.info(
                    "Epoch %d: unfreezing backbone — differential LR "
                    "(backbone=1e-5, head=%.0e). T_max=%d cosine epochs remain.",
                    epoch, HEAD_LR, EPOCHS - FREEZE_EPOCHS,
                )
                for param in model.parameters():
                    param.requires_grad = True
                backbone_params  = [
                    p for n, p in model.named_parameters()
                    if n not in head_param_names
                ]
                head_params_list = [
                    p for n, p in model.named_parameters()
                    if n in head_param_names
                ]
                optimizer = torch.optim.AdamW(
                    [
                        {"params": backbone_params,  "lr": 1e-5},
                        {"params": head_params_list, "lr": HEAD_LR},
                    ],
                    weight_decay=WEIGHT_DECAY,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=EPOCHS - FREEZE_EPOCHS,
                    eta_min=1e-6,
                )

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, torch.device(DEVICE)
            )
            val_loss, val_acc = evaluate_epoch(
                model, val_loader, criterion, torch.device(DEVICE)
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f  lr=%.2e",
                epoch, EPOCHS, train_loss, train_acc, val_loss, val_acc, current_lr,
            )

            if mlflow_active:
                import mlflow

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss":   val_loss,   "val_acc":   val_acc,
                        "lr":         current_lr,
                    },
                    step=epoch,
                )

            # ── Checkpoint selection and early stopping ────────────────────────
            if val_acc > best_val_acc:
                best_val_acc      = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), str(best_ckpt_path))
                logger.info(
                    "  ↑ New best val_acc=%.4f — saved to %s",
                    best_val_acc, best_ckpt_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d "
                        "(no val_acc improvement for %d consecutive epochs).",
                        epoch, EARLY_STOP_PATIENCE,
                    )
                    if mlflow_active:
                        import mlflow

                        mlflow.log_param("early_stop_epoch", epoch)
                    break

    finally:
        if mlflow_active:
            import mlflow

            mlflow.end_run()

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)
    logger.info("Checkpoint: %s", best_ckpt_path)
    logger.info("Next step:  python scripts/export_action_weights.py")


if __name__ == "__main__":
    main()
