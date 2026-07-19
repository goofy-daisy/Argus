from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Frame preprocessor for RGB and thermal modalities.

    RGB pipeline:
        BGR → RGB → resize 640×640 → [0, 1] → ImageNet mean/std normalisation.
        Output shape: (640, 640, 3), float32.

    Thermal pipeline:
        16-bit input → CLAHE contrast enhancement → 8-bit → [0, 1].
        Output shape: (H, W, 1), float32.

    Used by Phase 3 ingestion and Phases 4–8 downstream modules.
    """

    IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    YOLO_SIZE: tuple = (640, 640)

    def preprocess_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a BGR frame for YOLO inference.

        Pipeline: BGR→RGB → resize 640×640 → [0,1] → ImageNet normalisation.

        Args:
            frame: BGR uint8 array of shape (H, W, 3).

        Returns:
            float32 array of shape (640, 640, 3), ImageNet-normalised.
            Note: values are NOT bounded to [0,1] after ImageNet normalisation —
            they are centred around zero per channel.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.YOLO_SIZE, interpolation=cv2.INTER_LINEAR)
        unit = resized.astype(np.float32) / 255.0
        normalised = (unit - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return normalised

    def normalise_unit(self, frame: np.ndarray) -> np.ndarray:
        """Normalise a uint8 frame to [0.0, 1.0] float32.

        Does not apply ImageNet mean/std shift. Used for intermediate steps,
        thermal preprocessing, and visualisation contexts where [0,1] range
        must be preserved.

        Args:
            frame: uint8 numpy array of any shape.

        Returns:
            float32 array of the same shape, values in [0.0, 1.0].
        """
        return frame.astype(np.float32) / 255.0

    def preprocess_thermal(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a thermal frame via CLAHE contrast enhancement.

        Pipeline: 16-bit input → normalise to 8-bit → CLAHE → [0, 1].

        Args:
            frame: Thermal array, uint16 of shape (H, W) or (H, W, 1).

        Returns:
            float32 array of shape (H, W, 1), values in [0.0, 1.0].
        """
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        frame_8bit = cv2.normalize(
            frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(frame_8bit)

        normalised = self.normalise_unit(enhanced)
        return normalised[:, :, np.newaxis]
