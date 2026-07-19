from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from argus.ingestion.frame_batcher import FrameBatcher
from argus.ingestion.preprocessor import Preprocessor
from argus.ingestion.video_reader import VideoReader


# ── Synthetic data helpers ────────────────────────────────────────────────────

def make_video(
    path: str,
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
) -> None:
    """Write a synthetic MP4 to path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_image_sequence(
    directory: Path,
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
) -> None:
    """Write numbered JPEG frames to directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(directory / f"{i + 1:06d}.jpg"), frame)


# ── VideoReader — video file mode ─────────────────────────────────────────────

class TestVideoReaderVideoFile:

    def test_stride_1_yields_all_frames(self, tmp_path: Path) -> None:
        """stride=1 yields every frame."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=30)
        with VideoReader(path, stride=1) as reader:
            frames = list(reader.read_frames())
        assert len(frames) == 30

    def test_stride_3_yields_correct_count(self, tmp_path: Path) -> None:
        """stride=3 on 30 frames yields 10 frames."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=30)
        with VideoReader(path, stride=3) as reader:
            frames = list(reader.read_frames())
        assert len(frames) == 10

    def test_default_stride_is_3(self, tmp_path: Path) -> None:
        """Default stride is 3."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=30)
        with VideoReader(path) as reader:
            frames = list(reader.read_frames())
        assert len(frames) == 10

    def test_frame_shape_hwc(self, tmp_path: Path) -> None:
        """Yielded frames have shape (H, W, 3)."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=5, width=640, height=480)
        with VideoReader(path, stride=1) as reader:
            frames = list(reader.read_frames())
        assert frames[0].shape == (480, 640, 3)

    def test_frame_dtype_uint8(self, tmp_path: Path) -> None:
        """Yielded frames are uint8."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=5)
        with VideoReader(path, stride=1) as reader:
            frames = list(reader.read_frames())
        assert frames[0].dtype == np.uint8

    def test_processed_count_matches_yielded(self, tmp_path: Path) -> None:
        """processed_count matches the number of yielded frames."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=30)
        with VideoReader(path, stride=3) as reader:
            frames = list(reader.read_frames())
            assert reader.processed_count == len(frames) == 10

    def test_invalid_stride_raises(self, tmp_path: Path) -> None:
        """stride=0 raises ValueError."""
        path = str(tmp_path / "test.mp4")
        make_video(path, num_frames=5)
        with pytest.raises(ValueError):
            VideoReader(path, stride=0)


# ── VideoReader — image sequence mode ────────────────────────────────────────

class TestVideoReaderImageSequence:

    def test_image_sequence_frame_count(self, tmp_path: Path) -> None:
        """Image sequence with stride=3 on 30 frames yields 10 frames."""
        seq_dir = tmp_path / "img1"
        make_image_sequence(seq_dir, num_frames=30)
        with VideoReader(str(seq_dir), stride=3) as reader:
            frames = list(reader.read_frames())
        assert len(frames) == 10

    def test_image_sequence_frame_shape(self, tmp_path: Path) -> None:
        """Image sequence frames have correct (H, W, 3) shape."""
        seq_dir = tmp_path / "img1"
        make_image_sequence(seq_dir, num_frames=5, width=640, height=480)
        with VideoReader(str(seq_dir), stride=1) as reader:
            frames = list(reader.read_frames())
        assert frames[0].shape == (480, 640, 3)

    def test_image_sequence_dtype(self, tmp_path: Path) -> None:
        """Image sequence frames are uint8."""
        seq_dir = tmp_path / "img1"
        make_image_sequence(seq_dir, num_frames=5)
        with VideoReader(str(seq_dir), stride=1) as reader:
            frames = list(reader.read_frames())
        assert frames[0].dtype == np.uint8

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """Empty directory raises RuntimeError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError):
            VideoReader(str(empty_dir)).open()


# ── Preprocessor ─────────────────────────────────────────────────────────────

class TestPreprocessor:

    def setup_method(self) -> None:
        self.preprocessor = Preprocessor()

    def test_rgb_output_shape(self) -> None:
        """preprocess_rgb output is (640, 640, 3)."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess_rgb(frame)
        assert result.shape == (640, 640, 3)

    def test_rgb_dtype_float32(self) -> None:
        """preprocess_rgb output is float32."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess_rgb(frame)
        assert result.dtype == np.float32

    def test_normalise_unit_range(self) -> None:
        """normalise_unit output is in [0.0, 1.0]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = self.preprocessor.normalise_unit(frame)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_normalise_unit_dtype(self) -> None:
        """normalise_unit output is float32."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = self.preprocessor.normalise_unit(frame)
        assert result.dtype == np.float32

    def test_normalise_unit_zero_maps_to_zero(self) -> None:
        """A zero frame normalises to all zeros."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.preprocessor.normalise_unit(frame)
        assert float(result.max()) == 0.0

    def test_normalise_unit_max_maps_to_one(self) -> None:
        """A 255-filled frame normalises to all ones."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = self.preprocessor.normalise_unit(frame)
        assert abs(float(result.max()) - 1.0) < 1e-6

    def test_thermal_output_shape_2d_input(self) -> None:
        """preprocess_thermal on (H, W) input returns (H, W, 1)."""
        thermal = np.random.randint(0, 65536, (480, 640), dtype=np.uint16)
        result = self.preprocessor.preprocess_thermal(thermal)
        assert result.shape == (480, 640, 1)

    def test_thermal_output_shape_3d_input(self) -> None:
        """preprocess_thermal on (H, W, 1) input returns (H, W, 1)."""
        thermal = np.random.randint(0, 65536, (480, 640, 1), dtype=np.uint16)
        result = self.preprocessor.preprocess_thermal(thermal)
        assert result.shape == (480, 640, 1)

    def test_thermal_range(self) -> None:
        """preprocess_thermal output is in [0.0, 1.0]."""
        thermal = np.random.randint(0, 65536, (480, 640), dtype=np.uint16)
        result = self.preprocessor.preprocess_thermal(thermal)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_thermal_dtype(self) -> None:
        """preprocess_thermal output is float32."""
        thermal = np.random.randint(0, 65536, (480, 640), dtype=np.uint16)
        result = self.preprocessor.preprocess_thermal(thermal)
        assert result.dtype == np.float32


# ── FrameBatcher ─────────────────────────────────────────────────────────────

class TestFrameBatcher:

    def test_exact_multiple(self) -> None:
        """12 frames with batch_size=4 yields 3 full batches."""
        batcher = FrameBatcher(batch_size=4)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(12)]
        batches = list(batcher.batch(iter(frames)))
        assert len(batches) == 3
        assert all(len(b) == 4 for b in batches)

    def test_remainder_batch(self) -> None:
        """10 frames with batch_size=4 yields 2 full + 1 partial batch."""
        batcher = FrameBatcher(batch_size=4)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        batches = list(batcher.batch(iter(frames)))
        assert len(batches) == 3
        assert len(batches[-1]) == 2

    def test_batch_size_1(self) -> None:
        """batch_size=1 yields one frame per batch."""
        batcher = FrameBatcher(batch_size=1)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        batches = list(batcher.batch(iter(frames)))
        assert len(batches) == 5
        assert all(len(b) == 1 for b in batches)

    def test_empty_stream(self) -> None:
        """Empty input yields no batches."""
        batcher = FrameBatcher(batch_size=4)
        batches = list(batcher.batch(iter([])))
        assert len(batches) == 0

    def test_invalid_batch_size_zero(self) -> None:
        """batch_size=0 raises ValueError."""
        with pytest.raises(ValueError):
            FrameBatcher(batch_size=0)

    def test_invalid_batch_size_negative(self) -> None:
        """Negative batch_size raises ValueError."""
        with pytest.raises(ValueError):
            FrameBatcher(batch_size=-1)

    def test_total_frames_preserved(self) -> None:
        """Sum of all batch lengths equals total input frame count."""
        batcher = FrameBatcher(batch_size=7)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(23)]
        batches = list(batcher.batch(iter(frames)))
        assert sum(len(b) for b in batches) == 23
