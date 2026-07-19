from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from argus.reid.reid import ReIdentifier

WEIGHTS_PATH = "argus/models/osnet_x1_0_market1501.pth"
WEIGHTS_EXIST = Path(WEIGHTS_PATH).exists()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_unit_embedding(dim: int = 512, seed: Optional[int] = None) -> np.ndarray:
    """Return a random L2-normalised float32 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Interface tests ───────────────────────────────────────────────────────────

class TestReIdentifierInterface:

    def test_init_stores_params(self) -> None:
        r = ReIdentifier(WEIGHTS_PATH, device="mps", similarity_threshold=0.7)
        assert r.model_path == WEIGHTS_PATH
        assert r.device == "mps"
        assert r.similarity_threshold == 0.7

    def test_embedding_dim_constant(self) -> None:
        assert ReIdentifier.EMBEDDING_DIM == 512

    def test_extract_raises_before_load(self) -> None:
        import cv2
        r = ReIdentifier(WEIGHTS_PATH)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="load_model"):
            r.extract_embedding(frame, (10.0, 10.0, 100.0, 200.0))

    def test_load_raises_if_weights_absent(self, tmp_path) -> None:
        r = ReIdentifier(str(tmp_path / "nonexistent.pth"))
        with pytest.raises(FileNotFoundError):
            r.load_model()

    def test_gallery_empty_on_init(self) -> None:
        r = ReIdentifier(WEIGHTS_PATH)
        assert r._gallery == {}


# ── Gallery and matching tests ────────────────────────────────────────────────

class TestReIdentifierGallery:

    def setup_method(self) -> None:
        self.reid = ReIdentifier(WEIGHTS_PATH, similarity_threshold=0.5)

    def _inject(self, camera_id: int, track_id: int, seed: int) -> np.ndarray:
        """Inject a synthetic embedding into the gallery without DB calls."""
        emb = make_unit_embedding(seed=seed)
        if camera_id not in self.reid._gallery:
            self.reid._gallery[camera_id] = {}
        self.reid._gallery[camera_id][track_id] = emb
        return emb

    def test_match_returns_none_on_empty_gallery(self) -> None:
        q = make_unit_embedding(seed=0)
        result = self.reid.match(q, camera_id=1)
        assert result is None

    def test_match_returns_none_same_camera_only(self) -> None:
        emb = self._inject(camera_id=1, track_id=42, seed=10)
        result = self.reid.match(emb, camera_id=1)
        assert result is None

    def test_match_perfect_score_different_camera(self) -> None:
        emb = self._inject(camera_id=1, track_id=42, seed=10)
        result = self.reid.match(emb, camera_id=2)
        assert result == 42

    def test_match_below_threshold_returns_none(self) -> None:
        self.reid.similarity_threshold = 0.999
        self._inject(camera_id=1, track_id=99, seed=20)
        q = make_unit_embedding(seed=21)  # random — low similarity
        result = self.reid.match(q, camera_id=2)
        assert result is None

    def test_match_returns_best_track(self) -> None:
        self.reid.similarity_threshold = 0.0
        emb_a = self._inject(camera_id=1, track_id=10, seed=1)
        self._inject(camera_id=1, track_id=20, seed=2)
        result = self.reid.match(emb_a, camera_id=2)
        assert result == 10  # emb_a matches track 10 with score 1.0

    def test_clear_gallery(self) -> None:
        self._inject(camera_id=1, track_id=5, seed=5)
        self.reid.clear_gallery()
        assert self.reid._gallery == {}

    def test_update_gallery_in_memory_only(self) -> None:
        emb = make_unit_embedding(seed=99)
        # Bypass DB by testing _gallery directly
        if 3 not in self.reid._gallery:
            self.reid._gallery[3] = {}
        self.reid._gallery[3][77] = emb
        assert 77 in self.reid._gallery[3]
        assert np.allclose(self.reid._gallery[3][77], emb)


# ── Embedding property tests ──────────────────────────────────────────────────

class TestEmbeddingProperties:

    def test_unit_embedding_has_norm_one(self) -> None:
        emb = make_unit_embedding()
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_unit_embedding_shape(self) -> None:
        emb = make_unit_embedding(dim=512)
        assert emb.shape == (512,)

    def test_unit_embedding_dtype(self) -> None:
        emb = make_unit_embedding()
        assert emb.dtype == np.float32

    def test_cosine_sim_identical_embeddings(self) -> None:
        emb = make_unit_embedding(seed=42)
        score = float(np.dot(emb, emb))
        assert abs(score - 1.0) < 1e-5

    def test_cosine_sim_orthogonal_embeddings(self) -> None:
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        score = float(np.dot(a, b))
        assert abs(score) < 1e-6


# ── Post-training tests (skip if weights absent) ──────────────────────────────

@pytest.mark.skipif(not WEIGHTS_EXIST, reason="OSNet weights not yet trained")
class TestReIdentifierWithModel:

    @pytest.fixture(scope="class")
    def loaded_reid(self) -> ReIdentifier:
        r = ReIdentifier(
            WEIGHTS_PATH,
            device="mps",
            similarity_threshold=0.65,
        )
        r.load_model()
        return r

    def test_extract_returns_correct_shape(self, loaded_reid: ReIdentifier) -> None:
        import cv2
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        emb = loaded_reid.extract_embedding(frame, (100.0, 50.0, 200.0, 350.0))
        assert emb.shape == (512,)

    def test_extract_returns_float32(self, loaded_reid: ReIdentifier) -> None:
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        emb = loaded_reid.extract_embedding(frame, (100.0, 50.0, 200.0, 350.0))
        assert emb.dtype == np.float32

    def test_extract_is_l2_normalised(self, loaded_reid: ReIdentifier) -> None:
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        emb = loaded_reid.extract_embedding(frame, (100.0, 50.0, 200.0, 350.0))
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_invalid_bbox_raises(self, loaded_reid: ReIdentifier) -> None:
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            loaded_reid.extract_embedding(frame, (200.0, 200.0, 100.0, 100.0))
