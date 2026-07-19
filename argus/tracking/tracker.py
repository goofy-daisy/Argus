"""ByteTrack multi-object tracker for Argus.

Replaced the previous IoU+appearance tracker in Stage 1 of the upgrade plan.
Uses two-stage IoU association — high-confidence detections first, then
unmatched tracks re-matched against low-confidence detections — recovering
tracks that were discarded during occlusion. No appearance embedder;
all tracking is purely motion-based via Kalman filtering.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _DetResults:
    """Adapter that presents a numpy detection array as a BYTETracker-compatible results object.

    BYTETracker.update() expects an object with .conf, .cls, .xywh attributes and
    support for boolean indexing. This shim wraps our (N, 5) xyxyconf array and
    exposes exactly those attributes in the formats ByteTrack requires.
    """

    def __init__(self, xyxyconf: np.ndarray) -> None:
        """
        Args:
            xyxyconf: float32 array of shape (N, 5) — columns are [x1, y1, x2, y2, conf].
        """
        self._data = xyxyconf.astype(np.float32)
        n = len(self._data)
        if n > 0:
            x1 = self._data[:, 0]
            y1 = self._data[:, 1]
            x2 = self._data[:, 2]
            y2 = self._data[:, 3]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = x2 - x1
            h = y2 - y1
            self.xywh: np.ndarray = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)
        self.conf: np.ndarray = self._data[:, 4] if n > 0 else np.empty(0, dtype=np.float32)
        self.cls: np.ndarray = np.zeros(n, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: np.ndarray) -> _DetResults:
        """Support boolean or integer indexing, returning a new _DetResults."""
        return _DetResults(self._data[idx])


class _STrackProxy:
    """Thin adapter wrapping an ultralytics STrack to expose the Tracker public interface.

    Converts STrack's numpy-typed attributes to plain Python types and provides
    to_ltrb() / is_confirmed() in the contract expected by all Tracker callers.
    """

    def __init__(self, strack: object) -> None:
        self._strack = strack

    @property
    def track_id(self) -> int:
        return int(self._strack.track_id)  # type: ignore[attr-defined]

    def to_ltrb(self) -> Tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2) in pixel coordinates."""
        xyxy = self._strack.xyxy  # type: ignore[attr-defined]
        return (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

    def is_confirmed(self) -> bool:
        return bool(self._strack.is_activated)  # type: ignore[attr-defined]


class Tracker:
    """ByteTrack multi-object tracker.

    Wraps ultralytics BYTETracker with the same public interface that was
    previously exposed by the DeepSORT implementation. Two-stage IoU
    association recovers tracks occluded between frames without requiring
    any appearance embedder. Track IDs are native integers from ByteTrack.

    Upgraded from DeepSORT in Stage 1.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Args:
            config: Optional config dict. ByteTrack parameters are read from
                    its ``tracking`` block. All parameters fall back to safe
                    defaults if absent.
        """
        trk = (config or {}).get("tracking", {})

        self.track_thresh: float = float(trk.get("track_thresh", 0.5))
        self.track_low_thresh: float = float(trk.get("track_low_thresh", 0.1))
        self.new_track_thresh: float = float(trk.get("new_track_thresh", 0.6))
        self.track_buffer: int = int(trk.get("track_buffer", 30))
        self.match_thresh: float = float(trk.get("match_thresh", 0.8))

        self._tracker: Optional[object] = None
        self._frame_id: int = 0

    def _init_tracker(self) -> None:
        """Lazily initialise the BYTETracker instance on first update() call."""
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_high_thresh=self.track_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=False,
        )
        self._tracker = BYTETracker(args)
        logger.info(
            "BYTETracker initialised — track_thresh=%.2f, track_low_thresh=%.2f, "
            "new_track_thresh=%.2f, track_buffer=%d, match_thresh=%.2f",
            self.track_thresh,
            self.track_low_thresh,
            self.new_track_thresh,
            self.track_buffer,
            self.match_thresh,
        )

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float]],
        frame: np.ndarray,
    ) -> List[_STrackProxy]:
        """Update tracker state with new detections for the current frame.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) from Detector.
                        Pass an empty list for frames with no detections.
            frame: Current frame as numpy array (H, W, 3), BGR uint8.

        Returns:
            List of _STrackProxy objects for all confirmed tracks. Each exposes:
                .track_id  (int)  — persistent integer identity
                .to_ltrb() (tuple) — (x1, y1, x2, y2) bounding box as floats
                .is_confirmed() (bool) — always True for returned tracks
        """
        if self._tracker is None:
            self._init_tracker()

        self._frame_id += 1

        if len(detections) == 0:
            det_arr = np.empty((0, 5), dtype=np.float32)
        else:
            det_arr = np.array(detections, dtype=np.float32)

        det_results = _DetResults(det_arr)
        self._tracker.update(det_results, frame)  # type: ignore[union-attr]

        return [
            _STrackProxy(t)
            for t in self._tracker.tracked_stracks  # type: ignore[union-attr]
            if t.is_activated
        ]

    def reset(self) -> None:
        """Reset all tracker state. Call between independent video sequences."""
        self._tracker = None
        self._frame_id = 0
        logger.debug("Tracker reset.")
