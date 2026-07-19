from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Detector:
    """YOLOv8m-based person detector.

    Uses ultralytics YOLOv8m, filters to person class (COCO index 0),
    and runs inference on the configured device (default: mps).

    If the model file does not exist at model_path, ultralytics downloads
    it automatically to its local cache and a copy is saved to model_path
    for future runs.

    Implemented in Phase 4.
    """

    PERSON_CLASS_ID: int = 0

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        config: dict | None = None,
    ) -> None:
        """
        Args:
            model_path: Path to yolov8m.pt weights file. Downloads if absent.
            device: Inference device. "mps" for Apple Silicon, "cpu" as fallback.
            confidence_threshold: Minimum detection confidence. Detections below
                this value are discarded.
            config: Optional config dict. Values are read from its ``detection``
                block. Explicit arguments above take precedence over config,
                which in turn takes precedence over the hard defaults.
        """
        det = (config or {}).get("detection", {})

        self.model_path: str = (
            model_path if model_path is not None
            else det.get("model_path", "argus/models/yolov8m.pt")
        )
        self.device: str = (
            device if device is not None else det.get("device", "mps")
        )
        self.confidence_threshold: float = (
            confidence_threshold if confidence_threshold is not None
            else det.get("confidence_threshold", 0.10)
        )
        self.nms_iou_threshold: float = det.get("nms_iou_threshold", 0.45)
        self.imgsz: int = det.get("imgsz", 1280)
        self._model = None

    def load_model(self) -> None:
        """Load YOLOv8m weights and warm up with a dummy forward pass.

        Downloads weights via ultralytics if not present at model_path.
        Must be called before detect().
        """
        from ultralytics import YOLO

        path = Path(self.model_path)

        if path.exists():
            logger.info("Loading YOLOv8m from %s", self.model_path)
            self._model = YOLO(str(path))
        else:
            logger.info(
                "yolov8m.pt not found at %s — downloading via ultralytics",
                self.model_path,
            )
            self._model = YOLO("yolov8m.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            cached = Path.home() / ".ultralytics" / "assets" / "yolov8m.pt"
            if cached.exists() and not path.exists():
                shutil.copy(str(cached), str(path))
                logger.info("Saved YOLOv8m weights to %s", self.model_path)

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=self.device,
            classes=[0],
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        logger.info("YOLOv8m loaded and warmed up on device: %s", self.device)

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Detect persons in a single BGR frame.

        Args:
            frame: BGR uint8 numpy array of shape (H, W, 3).

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples in pixel coordinates.
            Only person-class detections above confidence_threshold are returned.
            Zero-area boxes are silently discarded.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Detector.detect() requires a raw uint8 BGR frame from VideoReader. "
                f"Received dtype={frame.dtype}. Do not apply Preprocessor.preprocess_rgb() "
                f"or any normalisation before calling detect(). "
                f"YOLOv8 handles its own preprocessing internally."
            )

        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before detect()."
            )

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.imgsz,
            classes=[0],
            verbose=False,
            device=self.device,
        )

        detections: List[Tuple[float, float, float, float, float]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                # Discard degenerate zero-area boxes — an ultralytics MPS
                # boundary artifact where a box at the frame edge has its
                # x2 clipped to equal x1 (width=0). Passing a zero-aspect-ratio
                # box into DeepSort corrupts its Kalman filter state.
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    logger.debug(
                        "Discarded degenerate box: x1=%.1f x2=%.1f y1=%.1f y2=%.1f conf=%.3f",
                        x1, x2, y1, y2, conf,
                    )
                    continue
                detections.append((x1, y1, x2, y2, conf))

        return detections