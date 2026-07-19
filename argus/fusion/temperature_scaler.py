from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Bins predictions by confidence. ECE is the weighted average of
    |bin_accuracy - bin_confidence| across all bins.

    Args:
        confidences: 1-D float array of predicted confidence scores in [0, 1].
        correct: 1-D binary array (1 = correct prediction, 0 = incorrect).
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE as float in [0, 1]. Lower is better.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    if len(confidences) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc  = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


class TemperatureScaler:
    """Post-hoc confidence calibration via temperature scaling.

    Divides model logits by a scalar temperature T, then re-applies
    softmax. T is optimised to minimise negative log-likelihood on a
    calibration set using PyTorch LBFGS.

    For binary detection confidence scores (not raw logits), use
    scale_confidence() which applies temperature via the logit transform:
        logit(conf) = log(conf / (1 - conf))
        scaled = sigmoid(logit(conf) / T)

    Usage:
        scaler = TemperatureScaler()
        scaler.fit_from_logits(logits, labels)         # multi-class
        # or
        scaler.fit_from_confidences(confidences, correct)  # binary
        cal_conf = scaler.scale_confidence(raw_conf)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """
        Args:
            temperature: Initial temperature. Overwritten after fit_*() calls.
                         T > 1 softens probabilities (reduces overconfidence).
                         T < 1 sharpens probabilities (increases confidence).
        """
        self.temperature: float = temperature

    def fit_from_logits(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Optimise T to minimise NLL on multi-class logits.

        Temperature is parameterised in log space (T = exp(log_T)) so
        gradients flow freely without a clamping cut-off inside LBFGS.

        Args:
            logits: (N, C) raw pre-softmax logits.
            labels: (N,) integer class labels in [0, C-1].

        Returns:
            Optimised temperature T.
        """
        import torch
        import torch.nn as nn

        logits_t = torch.FloatTensor(logits)
        labels_t = torch.LongTensor(labels)
        # Log-space: T = exp(log_T) is always positive; no gradient-blocking clamp needed
        log_T     = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            T    = torch.exp(log_T)
            loss = criterion(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        T_final = float(torch.exp(log_T).item())
        if not (0.1 <= T_final <= 10.0):
            logger.warning(
                "fit_from_logits: T=%.4f is outside [0.1, 10.0] — "
                "calibration may be suboptimal.",
                T_final,
            )
        self.temperature = float(max(0.1, min(10.0, T_final)))
        logger.info("Temperature fitted from logits: T=%.4f", self.temperature)
        return self.temperature

    def fit_from_confidences(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
    ) -> float:
        """Optimise T to minimise binary NLL on detection confidence scores.

        Treats each detection confidence as p(correct). Applies temperature
        via the logit transform and optimises for best binary calibration.
        Temperature is parameterised in log space (T = exp(log_T)) so
        gradients flow freely without a clamping cut-off inside LBFGS.

        Args:
            confidences: (N,) float array of confidence scores in (0, 1).
            correct: (N,) binary array — 1 if detection matched a GT bbox.

        Returns:
            Optimised temperature T.
        """
        import torch
        import torch.nn as nn

        eps        = 1e-6
        conf       = np.clip(confidences, eps, 1.0 - eps).astype(np.float32)
        logits_pos = np.log(conf / (1.0 - conf))
        logits_t   = torch.FloatTensor(
            np.stack([logits_pos, -logits_pos], axis=1)
        )
        labels_t = torch.LongTensor(correct.astype(int))

        # Log-space: T = exp(log_T) is always positive; no gradient-blocking clamp needed
        log_T     = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            T    = torch.exp(log_T)
            loss = criterion(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        T_final = float(torch.exp(log_T).item())
        if not (0.1 <= T_final <= 10.0):
            logger.warning(
                "fit_from_confidences: T=%.4f is outside [0.1, 10.0] — "
                "calibration may be suboptimal.",
                T_final,
            )
        self.temperature = float(max(0.1, min(10.0, T_final)))
        logger.info("Temperature fitted from confidences: T=%.4f", self.temperature)
        return self.temperature

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        """Divide multi-class logits by temperature.

        Args:
            logits: (..., C) float array of raw logits.

        Returns:
            Temperature-scaled logits, same shape.
        """
        return np.asarray(logits, dtype=np.float32) / self.temperature

    def scale_confidence(self, confidence: float) -> float:
        """Apply temperature scaling to a binary detection confidence score.

        Applies the logit transform, divides by T, then re-applies sigmoid.

        Args:
            confidence: Detection confidence in (0, 1).

        Returns:
            Calibrated confidence in (0, 1).
        """
        eps = 1e-6
        conf = float(np.clip(confidence, eps, 1.0 - eps))
        logit = np.log(conf / (1.0 - conf))
        scaled = float(logit) / self.temperature
        return float(1.0 / (1.0 + np.exp(-scaled)))

    def scale_confidences(self, confidences: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to an array of detection confidence scores.

        Args:
            confidences: (N,) float array of confidence scores in (0, 1).

        Returns:
            (N,) calibrated confidence scores.
        """
        eps = 1e-6
        conf = np.clip(confidences, eps, 1.0 - eps).astype(np.float64)
        logits = np.log(conf / (1.0 - conf))
        scaled = logits / self.temperature
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)
