from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from argus.action.action_classifier import ActionClassifier

WEIGHTS_PATH = "argus/models/x3d_s_argus.pth"
WEIGHTS_EXIST = Path(WEIGHTS_PATH).exists()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_frames(n: int = 16, h: int = 480, w: int = 640) -> List[np.ndarray]:
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ── Interface tests ───────────────────────────────────────────────────────────

class TestActionClassifierInterface:

    def test_init_stores_params(self) -> None:
        ac = ActionClassifier(WEIGHTS_PATH, device="mps", num_classes=5)
        assert ac.model_path == WEIGHTS_PATH
        assert ac.device == "mps"
        assert ac.num_classes == 5

    def test_labels_count(self) -> None:
        assert len(ActionClassifier.LABELS) == 5

    def test_labels_values(self) -> None:
        expected = {"normal", "loitering", "running", "falling", "crowd_formation"}
        assert set(ActionClassifier.LABELS) == expected

    def test_label_to_idx_consistent(self) -> None:
        for idx, label in enumerate(ActionClassifier.LABELS):
            assert ActionClassifier.LABEL_TO_IDX[label] == idx

    def test_invalid_num_classes_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionClassifier(WEIGHTS_PATH, num_classes=10)

    def test_classify_raises_before_load(self) -> None:
        ac = ActionClassifier(WEIGHTS_PATH)
        clip = np.zeros((3, 16, 182, 182), dtype=np.float32)
        with pytest.raises(RuntimeError, match="load_model"):
            ac.classify(clip)

    def test_load_raises_if_weights_absent(self, tmp_path) -> None:
        ac = ActionClassifier(str(tmp_path / "missing.pth"))
        with pytest.raises(FileNotFoundError):
            ac.load_model()


# ── Clip extraction tests (no model needed) ───────────────────────────────────

class TestExtractClip:

    def setup_method(self) -> None:
        self.ac = ActionClassifier(WEIGHTS_PATH)

    def test_output_shape(self) -> None:
        frames = make_frames(16)
        clip = self.ac.extract_clip(frames, (100.0, 50.0, 300.0, 400.0))
        assert clip.shape == (3, 16, 182, 182)

    def test_output_dtype_float32(self) -> None:
        frames = make_frames(16)
        clip = self.ac.extract_clip(frames, (50.0, 50.0, 200.0, 300.0))
        assert clip.dtype == np.float32

    def test_wrong_frame_count_raises(self) -> None:
        frames = make_frames(8)  # not 16
        with pytest.raises(ValueError, match="Expected 16"):
            self.ac.extract_clip(frames, (0.0, 0.0, 100.0, 100.0))

    def test_full_frame_bbox(self) -> None:
        frames = make_frames(16, h=480, w=640)
        clip = self.ac.extract_clip(frames, (0.0, 0.0, 640.0, 480.0))
        assert clip.shape == (3, 16, 182, 182)

    def test_clip_order_ctw(self) -> None:
        # C first, then T, then H, W
        frames = make_frames(16)
        clip = self.ac.extract_clip(frames, (10.0, 10.0, 200.0, 300.0))
        c, t, h, w = clip.shape
        assert c == 3
        assert t == 16
        assert h == 182
        assert w == 182

    def test_degenerate_bbox_handled(self) -> None:
        # Out-of-bounds bbox should not raise
        frames = make_frames(16, h=100, w=100)
        clip = self.ac.extract_clip(frames, (-10.0, -10.0, 500.0, 500.0))
        assert clip.shape == (3, 16, 182, 182)

    def test_different_frame_sizes(self) -> None:
        frames = make_frames(16, h=1080, w=1920)
        clip = self.ac.extract_clip(frames, (500.0, 200.0, 900.0, 800.0))
        assert clip.shape == (3, 16, 182, 182)


# ── Post-training tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(not WEIGHTS_EXIST, reason="X3D-S weights not yet trained")
class TestActionClassifierWithModel:

    @pytest.fixture(scope="class")
    def loaded_ac(self) -> ActionClassifier:
        ac = ActionClassifier(
            model_path=WEIGHTS_PATH,
            device="mps",
            num_classes=5,
            clip_frames=16,
            clip_size=182,
        )
        ac.load_model()
        return ac

    def test_classify_returns_tuple(self, loaded_ac: ActionClassifier) -> None:
        clip = np.random.randn(3, 16, 182, 182).astype(np.float32)
        result = loaded_ac.classify(clip)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_classify_label_valid(self, loaded_ac: ActionClassifier) -> None:
        clip = np.random.randn(3, 16, 182, 182).astype(np.float32)
        label, _ = loaded_ac.classify(clip)
        assert label in ActionClassifier.LABELS

    def test_classify_confidence_range(self, loaded_ac: ActionClassifier) -> None:
        clip = np.random.randn(3, 16, 182, 182).astype(np.float32)
        _, confidence = loaded_ac.classify(clip)
        assert 0.0 <= confidence <= 1.0

    def test_classify_same_clip_deterministic(self, loaded_ac: ActionClassifier) -> None:
        clip = np.random.randn(3, 16, 182, 182).astype(np.float32)
        r1 = loaded_ac.classify(clip)
        r2 = loaded_ac.classify(clip)
        assert r1 == r2
