from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _build_lstm_autoencoder(input_size: int, hidden_size: int, num_layers: int):
    """Build and return the LSTM autoencoder nn.Module."""
    import torch.nn as nn

    class _LSTMSeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            dropout = 0.1 if num_layers > 1 else 0.0
            self.seq_len = None   # set after build
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.output_proj = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            batch, seq, _ = x.shape
            _, (hidden, cell) = self.encoder(x)
            # context: last layer hidden, repeated seq_len times
            context = hidden[-1].unsqueeze(1).expand(batch, seq, hidden_size)
            decoded, _ = self.decoder(context.contiguous(), (hidden, cell))
            return self.output_proj(decoded)   # (batch, seq_len, input_size)

    return _LSTMSeq2Seq()


# ── AnomalyDetector ────────────────────────────────────────────────────────


class AnomalyDetector:
    """LSTM Autoencoder for trajectory anomaly detection.

    Trained on (x_norm, y_norm, vx, vy, aspect_ratio) sequences of
    length 30 from MOT17 confirmed tracks. Anomaly score is the MSE
    reconstruction error. Detection threshold is mean + 3*std of the
    validation set, stored in the model checkpoint and loaded at init.

    Input features are standardised using training-set statistics
    saved in the checkpoint before being passed to the model.

    Implemented in Phase 7.
    """

    FEATURE_NAMES = ["x_norm", "y_norm", "vx", "vy", "aspect_ratio"]

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        threshold: float = 0.05,
        sequence_length: int = 30,
        feature_dim: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        """
        Args:
            model_path: Path to trained LSTM autoencoder checkpoint.
            device: Inference device.
            threshold: Anomaly score threshold. Overwritten by checkpoint
                       value when load_model() is called.
            sequence_length: Number of frames per trajectory sequence.
            feature_dim: Number of features per frame. Must be 5.
            hidden_size: LSTM hidden units.
            num_layers: Number of LSTM layers in encoder and decoder.
        """
        self.model_path      = model_path
        self.device          = device
        self.threshold       = threshold
        self.sequence_length = sequence_length
        self.feature_dim     = feature_dim
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers

        self._model     = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std:  np.ndarray | None = None

    # ── Public interface ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load LSTM autoencoder weights and calibration stats from checkpoint.

        The checkpoint contains: state_dict, threshold, feat_mean, feat_std.
        threshold overrides the constructor value after loading.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        import torch

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"LSTM autoencoder weights not found at '{self.model_path}'. "
                "Run scripts/train_anomaly.py first."
            )

        checkpoint = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )

        model = _build_lstm_autoencoder(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(torch.device(self.device))
        model.eval()

        self._model     = model
        self.threshold  = float(checkpoint["threshold"])
        self._feat_mean = np.array(checkpoint["feat_mean"], dtype=np.float32)
        self._feat_std  = np.array(checkpoint["feat_std"],  dtype=np.float32)

        logger.info(
            "LSTM autoencoder loaded — threshold=%.6f device=%s",
            self.threshold, self.device,
        )

    def score(self, trajectory: np.ndarray) -> float:
        """Compute reconstruction MSE for a trajectory sequence.

        The trajectory is standardised using training-set statistics
        before being passed to the model.

        Args:
            trajectory: numpy float32 array of shape (30, 5) representing
                        (x_norm, y_norm, vx, vy, aspect_ratio) per frame.

        Returns:
            Reconstruction MSE as float. Higher values indicate more
            anomalous trajectories.

        Raises:
            RuntimeError: If load_model() has not been called.
            ValueError: If trajectory shape is not (sequence_length, feature_dim).
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before score()."
            )

        expected = (self.sequence_length, self.feature_dim)
        if trajectory.shape != expected:
            raise ValueError(
                f"Expected trajectory shape {expected}, got {trajectory.shape}."
            )

        import torch

        # Standardise with training statistics
        normed = (trajectory - self._feat_mean) / (self._feat_std + 1e-8)

        tensor = torch.FloatTensor(normed).unsqueeze(0)   # (1, 30, 5)
        tensor = tensor.to(torch.device(self.device))

        with torch.no_grad():
            reconstruction = self._model(tensor)

        mse = float(((reconstruction - tensor) ** 2).mean().cpu().numpy())
        return mse

    def is_anomalous(
        self,
        trajectory: np.ndarray,
    ) -> Tuple[bool, float]:
        """Determine whether a trajectory is anomalous.

        Args:
            trajectory: numpy float32 array of shape (30, 5).

        Returns:
            Tuple of (is_anomalous: bool, score: float).
            is_anomalous is True when score > threshold.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        s = self.score(trajectory)
        return s > self.threshold, s

    def train(self, sequences: np.ndarray) -> None:
        """Not implemented as a direct method.

        Training is performed via scripts/train_anomaly.py which handles
        data loading, standardisation, LSTM training, threshold calibration,
        and checkpoint saving.

        Raises:
            NotImplementedError: Always. Use scripts/train_anomaly.py.
        """
        raise NotImplementedError(
            "Direct training not supported. Use scripts/train_anomaly.py."
        )
