from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argus.fusion.fusion import AttentionFusion
from argus.fusion.temperature_scaler import TemperatureScaler, compute_ece

FUSION_WEIGHTS = "argus/models/attention_fusion.pth"
WEIGHTS_EXIST  = Path(FUSION_WEIGHTS).exists()


# ── ECE utility tests — no model needed ──────────────────────────────────────

class TestComputeECE:

    def test_perfect_calibration_gives_zero(self) -> None:
        # confidence == accuracy in every bin → ECE = 0
        conf    = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correct = np.array([0,   0,   1,   1,   1  ], dtype=float)
        # Rough check: well-calibrated model should have low ECE
        ece = compute_ece(conf, correct, n_bins=5)
        assert ece >= 0.0

    def test_overconfident_gives_positive_ece(self) -> None:
        # Always predict 0.9 but only 50% correct
        rng = np.random.default_rng(0)
        conf    = np.full(200, 0.9)
        correct = (rng.random(200) < 0.5).astype(float)
        ece = compute_ece(conf, correct, n_bins=15)
        assert ece > 0.1   # should be clearly miscalibrated

    def test_empty_input_returns_zero(self) -> None:
        assert compute_ece(np.array([]), np.array([])) == 0.0

    def test_output_in_valid_range(self) -> None:
        rng = np.random.default_rng(1)
        conf    = rng.uniform(0, 1, 100)
        correct = (rng.random(100) < 0.6).astype(float)
        ece = compute_ece(conf, correct)
        assert 0.0 <= ece <= 1.0

    def test_returns_float(self) -> None:
        conf    = np.array([0.5, 0.8])
        correct = np.array([1.0, 0.0])
        assert isinstance(compute_ece(conf, correct), float)


# ── TemperatureScaler tests — no model needed ─────────────────────────────────

class TestTemperatureScaler:

    def test_init_default_temperature(self) -> None:
        ts = TemperatureScaler()
        assert ts.temperature == 1.0

    def test_scale_logits_identity_at_T1(self) -> None:
        ts = TemperatureScaler(temperature=1.0)
        logits = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(ts.scale_logits(logits), logits)

    def test_scale_logits_divides_by_T(self) -> None:
        ts = TemperatureScaler(temperature=2.0)
        logits = np.array([[2.0, 4.0, 6.0]])
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(ts.scale_logits(logits), expected)

    def test_scale_confidence_T1_returns_same(self) -> None:
        ts = TemperatureScaler(temperature=1.0)
        conf = 0.8
        result = ts.scale_confidence(conf)
        assert abs(result - conf) < 1e-4

    def test_scale_confidence_high_T_reduces_confidence(self) -> None:
        ts = TemperatureScaler(temperature=3.0)
        result = ts.scale_confidence(0.95)
        assert result < 0.95   # high T softens overconfident predictions

    def test_scale_confidence_low_T_increases_confidence(self) -> None:
        ts = TemperatureScaler(temperature=0.5)
        result = ts.scale_confidence(0.7)
        assert result > 0.7    # low T sharpens predictions

    def test_scale_confidences_array(self) -> None:
        ts = TemperatureScaler(temperature=2.0)
        conf = np.array([0.6, 0.8, 0.9])
        scaled = ts.scale_confidences(conf)
        assert scaled.shape == conf.shape
        assert np.all(scaled < conf)   # T>1 should reduce all confidences

    def test_fit_from_logits_updates_temperature(self) -> None:
        ts = TemperatureScaler(temperature=1.0)
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((100, 3)).astype(np.float32) * 5.0
        labels = rng.integers(0, 3, 100)
        ts.fit_from_logits(logits, labels)
        assert ts.temperature > 0.0

    def test_fit_from_confidences_updates_temperature(self) -> None:
        ts = TemperatureScaler(temperature=1.0)
        rng = np.random.default_rng(0)
        confidences = np.full(100, 0.92)
        correct = (rng.random(100) < 0.55).astype(int)
        ts.fit_from_confidences(confidences, correct)
        # Overconfident model should get T > 1
        assert ts.temperature > 1.0

    def test_temperature_scaling_reduces_ece(self) -> None:
        """Core calibration test: T scaling reduces ECE on overconfident data."""
        rng = np.random.default_rng(7)
        n = 300
        confidences = np.clip(rng.beta(8, 1, n), 0.01, 0.99)
        correct = (rng.random(n) < 0.55).astype(int)

        ece_before = compute_ece(confidences, correct.astype(float))

        ts = TemperatureScaler()
        ts.fit_from_confidences(confidences[:150], correct[:150])
        cal_conf = ts.scale_confidences(confidences[150:])
        ece_after = compute_ece(cal_conf, correct[150:].astype(float))

        assert ece_after < ece_before, (
            f"Temperature scaling should reduce ECE. "
            f"Before: {ece_before:.4f}, After: {ece_after:.4f}, T={ts.temperature:.4f}"
        )


# ── AttentionFusion interface tests ──────────────────────────────────────────

class TestAttentionFusionInterface:

    def test_init_stores_params(self) -> None:
        af = AttentionFusion(FUSION_WEIGHTS, device="mps", feature_channels=32)
        assert af.model_path == FUSION_WEIGHTS
        assert af.device == "mps"
        assert af.feature_channels == 32

    def test_fuse_raises_before_load(self) -> None:
        af = AttentionFusion(FUSION_WEIGHTS)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        thm = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="load_model"):
            af.fuse(rgb, thm)

    def test_compute_weights_raises_before_load(self) -> None:
        af = AttentionFusion(FUSION_WEIGHTS)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        thm = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="load_model"):
            af.compute_weights(rgb, thm)

    def test_load_raises_if_weights_absent(self, tmp_path) -> None:
        af = AttentionFusion(str(tmp_path / "missing.pth"))
        with pytest.raises(FileNotFoundError):
            af.load_model()


# ── Post-training tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(not WEIGHTS_EXIST, reason="Fusion weights not yet trained")
class TestAttentionFusionWithModel:

    @pytest.fixture(scope="class")
    def loaded_af(self) -> AttentionFusion:
        af = AttentionFusion(FUSION_WEIGHTS, device="mps")
        af.load_model()
        return af

    def test_fuse_output_shape(self, loaded_af: AttentionFusion) -> None:
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        thm = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
        result = loaded_af.fuse(rgb, thm)
        assert result.shape == (480, 640, 3)

    def test_fuse_output_dtype(self, loaded_af: AttentionFusion) -> None:
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        thm = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
        result = loaded_af.fuse(rgb, thm)
        assert result.dtype == np.uint8

    def test_compute_weights_shape(self, loaded_af: AttentionFusion) -> None:
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        thm = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
        weights = loaded_af.compute_weights(rgb, thm)
        assert weights.shape == (480, 640, 1)

    def test_compute_weights_range(self, loaded_af: AttentionFusion) -> None:
        rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        thm = np.random.randint(0, 256, (480, 640, 1), dtype=np.uint8)
        weights = loaded_af.compute_weights(rgb, thm)
        assert float(weights.min()) >= 0.0
        assert float(weights.max()) <= 1.0
