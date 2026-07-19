from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ActionClassifier:
    """X3D-S temporal action recognition module.

    Classifies behaviour over 16-frame clips extracted from active tracks.
    Label set: normal, loitering, running, falling, crowd_formation.
    Uses sliding window inference every 8 frames with majority vote over
    the last 3 predictions for temporal stability.

    X3D-S is loaded via pytorchvideo. Kinetics-400 pretrained weights are
    used as the backbone for fine-tuning on UCF101 proxy classes.

    Implemented in Phase 6.
    """

    LABELS: List[str] = [
        "normal",
        "loitering",
        "running",
        "falling",
        "crowd_formation",
    ]
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        num_classes: int = 5,
        clip_frames: int = 16,
        clip_size: int = 182,
        video_mean: Optional[List[float]] = None,
        video_std: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            model_path: Path to trained X3D-S weights (.pth file).
            device: Inference device. "mps" for Apple Silicon.
            num_classes: Number of output classes. Must equal len(LABELS).
            clip_frames: Number of frames per clip. Must be 16.
            clip_size: Spatial size (height and width) of each frame crop.
            video_mean: Per-channel normalisation mean (RGB order).
            video_std: Per-channel normalisation std (RGB order).

        Raises:
            ValueError: If num_classes != len(LABELS).
        """
        if num_classes != len(self.LABELS):
            raise ValueError(
                f"num_classes={num_classes} does not match "
                f"len(LABELS)={len(self.LABELS)}"
            )
        self.model_path = model_path
        self.device = device
        self.num_classes = num_classes
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.video_mean = np.array(
            video_mean if video_mean else [0.45, 0.45, 0.45],
            dtype=np.float32,
        )
        self.video_std = np.array(
            video_std if video_std else [0.225, 0.225, 0.225],
            dtype=np.float32,
        )
        self._model = None

    # ── Public interface ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load X3D-S architecture and fine-tuned weights.

        Builds the X3D-S model via pytorchvideo (architecture only, no
        pretrained weights). Replaces the Kinetics-400 classification head
        with a num_classes head. Loads the Argus-trained state dict.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        import torch
        import torch.nn as nn

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"X3D-S weights not found at '{self.model_path}'. "
                "Run scripts/train_action.py first."
            )

        logger.info("Loading X3D-S architecture from pytorchvideo...")
        # Load architecture without pretrained weights (we load our own below)
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_s",
            pretrained=False,
        )

        # Replace classification head: find last Linear layer and replace it
        self._replace_head(model, self.num_classes, nn)
        logger.info("Classification head replaced with %d-class output.", self.num_classes)

        # Load Argus-trained weights
        state_dict = torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded X3D-S weights from %s", self.model_path)

        model.to(torch.device(self.device))
        model.eval()
        self._model = model
        logger.info("X3D-S ready on device: %s", self.device)

    def extract_clip(
        self,
        frames: List[np.ndarray],
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop and preprocess 16 BGR frames into an X3D-S input tensor.

        Crops the bounding box region from each frame, converts BGR→RGB,
        resizes to clip_size×clip_size, and applies video normalisation.
        Does not require load_model() to be called.

        Args:
            frames: List of exactly clip_frames (16) BGR uint8 numpy arrays
                    of shape (H, W, 3).
            bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            numpy float32 array of shape (3, clip_frames, clip_size, clip_size)
            in (C, T, H, W) order, video-normalised.

        Raises:
            ValueError: If len(frames) != clip_frames.
        """
        if len(frames) != self.clip_frames:
            raise ValueError(
                f"Expected {self.clip_frames} frames, got {len(frames)}."
            )

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        clip: List[np.ndarray] = []
        for frame in frames:
            h, w = frame.shape[:2]
            fx1 = max(0, min(x1, w - 1))
            fy1 = max(0, min(y1, h - 1))
            fx2 = max(fx1 + 1, min(x2, w))
            fy2 = max(fy1 + 1, min(y2, h))

            crop = frame[fy1:fy2, fx1:fx2]
            if crop.size == 0:
                crop = np.zeros((self.clip_size, self.clip_size, 3), dtype=np.uint8)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                crop_rgb,
                (self.clip_size, self.clip_size),
                interpolation=cv2.INTER_LINEAR,
            )
            clip.append(resized)

        # Stack: (T, H, W, 3) → float, normalise
        clip_arr = np.stack(clip, axis=0).astype(np.float32) / 255.0
        clip_arr = (clip_arr - self.video_mean) / self.video_std

        # Transpose (T, H, W, C) → (C, T, H, W)
        clip_arr = clip_arr.transpose(3, 0, 1, 2).astype(np.float32)
        return clip_arr

    def classify(
        self,
        clip: np.ndarray,
    ) -> Tuple[str, float]:
        """Run X3D-S inference on a preprocessed clip.

        Args:
            clip: numpy float32 array of shape (3, clip_frames, clip_size, clip_size).

        Returns:
            Tuple of (label_string, confidence_float), e.g. ("loitering", 0.87).

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before classify()."
            )

        import torch

        tensor = torch.from_numpy(clip).unsqueeze(0)  # (1, C, T, H, W)
        tensor = tensor.to(torch.device(self.device))

        with torch.no_grad():
            logits = self._model(tensor)

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.LABELS[idx]

        return label, confidence

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _replace_head(model, num_classes: int, nn) -> None:
        """Replace the final Linear layer with a num_classes head.

        Iterates named modules to find the last nn.Linear layer and replaces
        it in-place. Also replaces any final Sigmoid/Softmax with Identity
        so that classify() receives raw logits.
        """
        last_name: Optional[str] = None
        last_module: Optional[object] = None

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_name = name
                last_module = module

        if last_name is None:
            raise RuntimeError("No Linear layer found in model to replace.")

        in_features = last_module.in_features  # type: ignore[attr-defined]
        parts = last_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        attr = parts[-1]
        setattr(parent, attr, nn.Linear(in_features, num_classes, bias=True))

        # Replace any final sigmoid/softmax activation with Identity
        for name, module in model.named_modules():
            if isinstance(module, (nn.Sigmoid, nn.Softmax)):
                p_parts = name.split(".")
                p_parent = model
                for part in p_parts[:-1]:
                    p_parent = p_parent[int(part)] if part.isdigit() else getattr(p_parent, part)
                setattr(p_parent, p_parts[-1], nn.Identity())
