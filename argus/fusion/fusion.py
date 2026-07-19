from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Fusion network factory ─────────────────────────────────────────────────


def _build_fusion_net(feature_channels: int = 32):
    """Build and return the attention gate fusion nn.Module.

    Architecture:
        RGB branch:     3  → feature_channels (2-layer CNN)
        Thermal branch: 1  → feature_channels (2-layer CNN)
        Gate:           2*feature_channels → 1 (sigmoid spatial weight)
        Fusion:         gate * thermal_feats + (1-gate) * rgb_feats
        Output:         feature_channels → 3 (1×1 projection for detector)
    """
    import torch.nn as nn

    C = feature_channels

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb_enc = nn.Sequential(
                nn.Conv2d(3, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
            )
            self.thm_enc = nn.Sequential(
                nn.Conv2d(1, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
            )
            self.gate = nn.Sequential(
                nn.Conv2d(C * 2, C, 1), nn.ReLU(inplace=True),
                nn.Conv2d(C, 1, 1),     nn.Sigmoid(),
            )
            self.out_proj = nn.Conv2d(C, 3, 1)

        def forward(self, rgb, thm):
            # rgb: (B, 3, H, W)  thm: (B, 1, H, W)
            import torch
            rgb_f = self.rgb_enc(rgb)
            thm_f = self.thm_enc(thm)
            w = self.gate(torch.cat([rgb_f, thm_f], dim=1))   # (B,1,H,W) sigmoid
            fused = w * thm_f + (1.0 - w) * rgb_f
            return self.out_proj(fused), w                      # (B,3,H,W), (B,1,H,W)

    return FusionNet()


# ── AttentionFusion ────────────────────────────────────────────────────────


class AttentionFusion:
    """Thermal/RGB attention gate fusion module.

    Learns per-spatial-location sigmoid weights to combine thermal and RGB
    feature streams. Outperforms channel concatenation by allowing the gate
    to learn when thermal dominates (low-light) versus RGB (daylight) per
    spatial region.

    The fuse() output is a 3-channel BGR uint8 array that can be passed
    directly to Detector.detect() as a fused frame.

    Implemented in Phase 8.
    """

    # ImageNet normalisation applied to RGB input
    RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        feature_channels: int = 32,
    ) -> None:
        """
        Args:
            model_path: Path to trained attention gate weights (.pth).
            device: Inference device.
            feature_channels: Number of intermediate feature channels. Must
                              match the value used during training (default 32).
        """
        self.model_path      = model_path
        self.device          = device
        self.feature_channels = feature_channels
        self._model = None

    # ── Public interface ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load attention gate weights.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        import torch

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Attention fusion weights not found at '{self.model_path}'. "
                "Run scripts/train_fusion.py first."
            )

        model = _build_fusion_net(feature_channels=self.feature_channels)
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(torch.device(self.device))
        model.eval()
        self._model = model
        logger.info("Attention fusion loaded on %s", self.device)

    def fuse(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
    ) -> np.ndarray:
        """Fuse RGB and thermal frames using the learned attention gate.

        Args:
            rgb_frame: BGR uint8 array of shape (H, W, 3).
            thermal_frame: Grayscale uint8 array of shape (H, W, 1)
                           or (H, W). CLAHE-normalised 8-bit thermal.

        Returns:
            Fused BGR uint8 array of shape (H, W, 3), ready for
            Detector.detect().

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before fuse()."
            )

        import torch

        h, w = rgb_frame.shape[:2]

        # Preprocess RGB: BGR → RGB → float → ImageNet norm → (1,3,H,W)
        rgb_rgb   = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_rgb.astype(np.float32) / 255.0
        rgb_norm  = (rgb_float - self.RGB_MEAN) / self.RGB_STD
        rgb_t     = torch.FloatTensor(rgb_norm.transpose(2, 0, 1)).unsqueeze(0)

        # Preprocess thermal: float → [0,1] → (1,1,H,W)
        if thermal_frame.ndim == 2:
            thermal_frame = thermal_frame[:, :, np.newaxis]
        thm_float = thermal_frame.astype(np.float32) / 255.0
        thm_t     = torch.FloatTensor(thm_float.transpose(2, 0, 1)).unsqueeze(0)

        rgb_t = rgb_t.to(torch.device(self.device))
        thm_t = thm_t.to(torch.device(self.device))

        with torch.no_grad():
            fused_t, _ = self._model(rgb_t, thm_t)   # (1, 3, H, W)

        # Postprocess: de-normalise → [0,1] → uint8 BGR
        fused_np = fused_t[0].cpu().numpy().transpose(1, 2, 0)
        fused_np = fused_np * self.RGB_STD + self.RGB_MEAN
        fused_np = np.clip(fused_np, 0.0, 1.0)
        fused_rgb = (fused_np * 255.0).astype(np.uint8)
        return cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR)

    def compute_weights(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
    ) -> np.ndarray:
        """Compute per-spatial-location sigmoid attention weights.

        Args:
            rgb_frame: BGR uint8 array of shape (H, W, 3).
            thermal_frame: Grayscale uint8 array of shape (H, W, 1) or (H, W).

        Returns:
            Sigmoid weight map as float32 array of shape (H, W, 1),
            values in [0, 1]. Values near 1 mean thermal dominates;
            values near 0 mean RGB dominates.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before compute_weights()."
            )

        import torch

        rgb_rgb   = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        rgb_float = rgb_rgb.astype(np.float32) / 255.0
        rgb_norm  = (rgb_float - self.RGB_MEAN) / self.RGB_STD
        rgb_t     = torch.FloatTensor(rgb_norm.transpose(2, 0, 1)).unsqueeze(0)

        if thermal_frame.ndim == 2:
            thermal_frame = thermal_frame[:, :, np.newaxis]
        thm_t = torch.FloatTensor(
            (thermal_frame.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0)

        rgb_t = rgb_t.to(torch.device(self.device))
        thm_t = thm_t.to(torch.device(self.device))

        with torch.no_grad():
            _, weights_t = self._model(rgb_t, thm_t)   # (1, 1, H, W)

        weights = weights_t[0, 0].cpu().numpy()
        return weights[:, :, np.newaxis].astype(np.float32)
