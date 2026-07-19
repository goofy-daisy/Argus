from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from argus.detection.detector import Detector
from argus.tracking.tracker import Tracker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_detector() -> Detector:
    """Load YOLOv8m once for the entire test module."""
    d = Detector(
        model_path="argus/models/yolov8m.pt",
        device="mps",
        confidence_threshold=0.5,
    )
    d.load_model()
    return d


# ── Detector tests ────────────────────────────────────────────────────────────

class TestDetector:

    def test_init_stores_params(self) -> None:
        """Constructor correctly stores all parameters."""
        d = Detector("argus/models/yolov8m.pt", device="mps", confidence_threshold=0.4)
        assert d.model_path == "argus/models/yolov8m.pt"
        assert d.device == "mps"
        assert d.confidence_threshold == 0.4

    def test_person_class_id_is_zero(self) -> None:
        """COCO person class index is 0."""
        assert Detector.PERSON_CLASS_ID == 0

    def test_detect_raises_before_load(self) -> None:
        """detect() raises RuntimeError if load_model() not called."""
        d = Detector("argus/models/yolov8m.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="load_model"):
            d.detect(frame)

    def test_detect_returns_list(self, loaded_detector: Detector) -> None:
        """detect() always returns a list."""
        frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        assert isinstance(result, list)

    def test_detect_blank_frame_returns_empty(self, loaded_detector: Detector) -> None:
        """A fully black frame produces no person detections."""
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        assert result == []

    def test_detect_output_tuple_length(self, loaded_detector: Detector) -> None:
        """Each detection is a 5-tuple: (x1, y1, x2, y2, confidence)."""
        frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        for det in result:
            assert len(det) == 5

    def test_detect_bbox_coordinates_valid(self, loaded_detector: Detector) -> None:
        """x2 > x1 and y2 > y1 for all returned detections."""
        frame = np.random.randint(128, 255, (640, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        for (x1, y1, x2, y2, conf) in result:
            assert x2 > x1, "x2 must be greater than x1"
            assert y2 > y1, "y2 must be greater than y1"

    def test_detect_confidence_above_threshold(self, loaded_detector: Detector) -> None:
        """All returned confidences are >= confidence_threshold."""
        frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        result = loaded_detector.detect(frame)
        for (_, _, _, _, conf) in result:
            assert conf >= loaded_detector.confidence_threshold


# ── Tracker tests ─────────────────────────────────────────────────────────────

_BYTETRACK_CFG = {
    "tracking": {
        "track_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 30,
        "match_thresh": 0.8,
    }
}


class TestTracker:
    """All tests use ByteTrack via config — no appearance embedder, no model download."""

    def test_init_stores_params(self) -> None:
        """Constructor correctly stores all ByteTrack parameters from config."""
        t = Tracker(config=_BYTETRACK_CFG)
        assert t.track_thresh == 0.5
        assert t.match_thresh == 0.8
        assert t.track_buffer == 30

    def test_tracker_is_none_before_first_update(self) -> None:
        """Internal BYTETracker instance is None until first update() call."""
        t = Tracker(config=_BYTETRACK_CFG)
        assert t._tracker is None

    def test_update_empty_detections_returns_list(self) -> None:
        """update() with no detections returns an empty list."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = tracker.update([], frame)
        assert isinstance(result, list)

    def test_update_with_detections_returns_list(self) -> None:
        """update() with detections returns a list."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = [(100.0, 100.0, 200.0, 300.0, 0.9)]
        result = tracker.update(detections, frame)
        assert isinstance(result, list)

    def test_confirmed_track_after_min_hits(self) -> None:
        """A track becomes confirmed after consecutive detections."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = [(100.0, 100.0, 200.0, 300.0, 0.9)]

        confirmed = []
        for _ in range(5):
            confirmed = tracker.update(detections, frame)

        assert len(confirmed) > 0

    def test_confirmed_track_has_track_id(self) -> None:
        """Confirmed tracks expose an integer track_id attribute."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = [(100.0, 100.0, 200.0, 300.0, 0.9)]

        tracks = []
        for _ in range(5):
            tracks = tracker.update(detections, frame)

        if tracks:
            assert hasattr(tracks[0], "track_id")
            assert isinstance(tracks[0].track_id, int)

    def test_confirmed_track_has_to_ltrb(self) -> None:
        """Confirmed tracks expose a to_ltrb() method returning 4 coords."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = [(100.0, 100.0, 200.0, 300.0, 0.9)]

        tracks = []
        for _ in range(5):
            tracks = tracker.update(detections, frame)

        if tracks:
            bbox = tracks[0].to_ltrb()
            assert len(bbox) == 4

    def test_reset_clears_internal_tracker(self) -> None:
        """reset() sets _tracker back to None."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tracker.update([(100.0, 100.0, 200.0, 300.0, 0.9)], frame)
        assert tracker._tracker is not None

        tracker.reset()
        assert tracker._tracker is None

    def test_multiple_detections_handled(self) -> None:
        """Tracker handles multiple simultaneous detections without error."""
        tracker = Tracker(config=_BYTETRACK_CFG)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        detections = [
            (50.0, 50.0, 150.0, 250.0, 0.9),
            (300.0, 100.0, 400.0, 300.0, 0.8),
            (500.0, 200.0, 600.0, 400.0, 0.75),
        ]
        result = tracker.update(detections, frame)
        assert isinstance(result, list)
