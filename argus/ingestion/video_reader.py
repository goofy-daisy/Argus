from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """OpenCV-based reader for video files, RTSP streams, and image sequences.

    Supports MP4, AVI video files, RTSP stream URLs, and directories of
    sequentially named image files (e.g. MOT17 img1/ directories).

    Yields BGR numpy arrays of shape (H, W, 3) per sampled frame.
    Tracks processed frame count, dropped frames, and pipeline throughput.

    Usage:
        with VideoReader("path/to/video.mp4", stride=3) as reader:
            for frame in reader.read_frames():
                process(frame)
        print(reader.throughput_fps)
    """

    SUPPORTED_IMAGE_EXTENSIONS: frozenset = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp"}
    )

    def __init__(
        self,
        source: Union[str, Path],
        stride: int = 3,
    ) -> None:
        """
        Args:
            source: Path to a video file, RTSP URL, or directory of image frames.
            stride: Sample every nth frame. Default 3 (every 3rd frame).

        Raises:
            ValueError: If stride is less than 1.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.source: str = str(source)
        self.stride: int = stride

        self._cap: Optional[cv2.VideoCapture] = None
        self._image_files: Optional[List[Path]] = None
        self._is_image_sequence: bool = False

        self._processed_count: int = 0
        self._dropped_count: int = 0
        self._start_time: Optional[float] = None

    # ── Public interface ────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the video source.

        Raises:
            RuntimeError: If the source cannot be opened or contains no frames.
        """
        source_path = Path(self.source)

        if source_path.is_dir():
            self._open_image_sequence(source_path)
        else:
            self._open_video()

    def read_frames(self) -> Generator[np.ndarray, None, None]:
        """Yield BGR frames sampled at the configured stride.

        Must call open() first.

        Yields:
            numpy array of shape (H, W, 3), dtype uint8, BGR colour order.

        Raises:
            RuntimeError: If open() has not been called.
        """
        if self._cap is None and self._image_files is None:
            raise RuntimeError("Call open() before read_frames().")

        self._processed_count = 0
        self._dropped_count = 0
        self._start_time = time.perf_counter()

        if self._is_image_sequence:
            yield from self._read_image_sequence()
        else:
            yield from self._read_video()

        elapsed = time.perf_counter() - self._start_time
        fps = self._processed_count / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Read complete — yielded: %d, dropped: %d, throughput: %.1f fps",
            self._processed_count,
            self._dropped_count,
            fps,
        )

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def native_fps(self) -> float:
        """Native frame rate of the video source. Returns 0.0 for image sequences."""
        if self._cap is not None:
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        return 0.0

    @property
    def total_frames(self) -> int:
        """Total frame count in the source. Returns image file count for sequences."""
        if self._is_image_sequence and self._image_files is not None:
            return len(self._image_files)
        if self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    @property
    def processed_count(self) -> int:
        """Number of frames yielded by the most recent read_frames() call."""
        return self._processed_count

    @property
    def dropped_count(self) -> int:
        """Number of frames that failed to read in the most recent call."""
        return self._dropped_count

    @property
    def throughput_fps(self) -> float:
        """Measured throughput fps. Accurate only after read_frames() completes."""
        if self._start_time is None or self._processed_count == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._processed_count / elapsed if elapsed > 0 else 0.0

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _open_video(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        logger.info(
            "Opened video: %s  (native fps=%.1f, total frames=%d)",
            self.source,
            self.native_fps,
            self.total_frames,
        )

    def _open_image_sequence(self, directory: Path) -> None:
        self._image_files = sorted(
            f
            for f in directory.iterdir()
            if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
        )
        if not self._image_files:
            raise RuntimeError(
                f"No supported image files found in directory: {self.source}"
            )
        self._is_image_sequence = True
        logger.info(
            "Opened image sequence: %s  (%d files)",
            self.source,
            len(self._image_files),
        )

    def _read_video(self) -> Generator[np.ndarray, None, None]:
        frame_index = 0
        while True:
            ret, frame = self._cap.read()  # type: ignore[union-attr]
            if not ret:
                if frame_index > 0:
                    # Only count as dropped if we were mid-sequence
                    pass
                break
            if frame_index % self.stride == 0:
                self._processed_count += 1
                yield frame
            else:
                pass  # stride-skipped frame — not a drop
            frame_index += 1

    def _read_image_sequence(self) -> Generator[np.ndarray, None, None]:
        for idx, image_path in enumerate(self._image_files):  # type: ignore[union-attr]
            if idx % self.stride != 0:
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                self._dropped_count += 1
                logger.warning("Failed to read image: %s", image_path)
                continue
            self._processed_count += 1
            yield frame
