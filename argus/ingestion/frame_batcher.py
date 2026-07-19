from __future__ import annotations

from typing import Generator, Iterable, List

import numpy as np


class FrameBatcher:
    """Batches a stream of frames into fixed-size lists for GPU throughput.

    Yields List[np.ndarray] of length batch_size. The final batch may be
    shorter than batch_size if total frames are not evenly divisible.

    Usage:
        batcher = FrameBatcher(batch_size=8)
        for batch in batcher.batch(reader.read_frames()):
            tensor = torch.stack([preprocess(f) for f in batch])
            model(tensor)
    """

    def __init__(self, batch_size: int = 8) -> None:
        """
        Args:
            batch_size: Number of frames per batch. Must be >= 1.

        Raises:
            ValueError: If batch_size is less than 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size: int = batch_size

    def batch(
        self,
        frames: Iterable[np.ndarray],
    ) -> Generator[List[np.ndarray], None, None]:
        """Yield batches of frames from an iterable.

        Args:
            frames: Any iterable of numpy arrays (H, W, 3).

        Yields:
            List[np.ndarray] of length batch_size, except possibly the final
            batch which may be shorter.
        """
        current: List[np.ndarray] = []
        for frame in frames:
            current.append(frame)
            if len(current) == self.batch_size:
                yield current
                current = []
        if current:
            yield current
