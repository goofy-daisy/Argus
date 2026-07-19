from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ReIdentifier:
    """OSNet-x1.0 cross-camera re-identification module.

    Extracts 512-dim L2-normalised appearance embeddings via torchreid.
    Matches query embeddings against an in-memory gallery using cosine
    similarity. Persists embeddings to the PostgreSQL embeddings table.
    Resumes the original track_id when a match exceeds the configured
    similarity threshold.

    Gallery is keyed by camera_id → track_id → embedding array. Same-camera
    entries are excluded from match() to enforce cross-camera identity linking.

    Implemented in Phase 5.
    """

    EMBEDDING_DIM: int = 512
    GALLERY_ALPHA: float = 0.9   # EMA weight for gallery updates (0=replace, 1=keep)

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        similarity_threshold: float = 0.65,
        input_height: int = 256,
        input_width: int = 128,
        num_train_pids: int = 751,
        config: Optional[dict] = None,
    ) -> None:
        """
        Args:
            model_path: Path to OSNet .pth weights file.
            device: Inference device. "mps" for Apple Silicon.
            similarity_threshold: Minimum cosine similarity for a valid
                cross-camera identity match.
            input_height: Crop height fed to OSNet. Market-1501 standard is 256.
            input_width: Crop width fed to OSNet. Market-1501 standard is 128.
            num_train_pids: Number of training identities. Market-1501 = 751.
            config: Optional full system config dict. reid.model_name is read
                    from it; defaults to "osnet_ain_x1_0" when absent.
        """
        self.model_path = model_path
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.input_height = input_height
        self.input_width = input_width
        self.num_train_pids = num_train_pids
        # Model name is config-driven to allow switching between OSNet variants.
        self.model_name: str = (config or {}).get("reid", {}).get("model_name", "osnet_ain_x1_0")
        self._model = None
        self._transform = None
        # In-memory gallery: {camera_id: {track_id: embedding (512,)}}
        self._gallery: Dict[int, Dict[int, np.ndarray]] = {}

    # ── Public interface ────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load OSNet-x1.0 weights and prepare the preprocessing transform.

        Raises:
            FileNotFoundError: If the weights file does not exist at model_path.
            RuntimeError: If torchreid cannot build or load the model.
        """
        import torch
        import torchreid
        from torchvision import transforms

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"OSNet weights not found at '{self.model_path}'. "
                "Run scripts/train_reid.py first, then re-run this step."
            )

        logger.info("Building %s (num_classes=%d)...", self.model_name, self.num_train_pids)
        self._model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=self.num_train_pids,
            pretrained=False,
            use_gpu=False,   # device assignment handled manually below
        )
        torchreid.utils.load_pretrained_weights(self._model, str(path))
        logger.info("Loaded %s weights from %s", self.model_name, self.model_path)

        self._model.to(torch.device(self.device))
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_height, self.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("%s ready on device: %s", self.model_name, self.device)

    def extract_embedding(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Extract a 512-dim L2-normalised embedding for a person crop.

        Args:
            frame: Full BGR frame as numpy array (H, W, 3), uint8.
            bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            numpy array of shape (512,), dtype float32, L2-normalised.

        Raises:
            RuntimeError: If load_model() has not been called.
            ValueError: If the crop derived from bbox is empty.
        """
        if self._model is None or self._transform is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() before extract_embedding()."
            )

        import torch

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Bounding box {bbox} produces an empty crop on frame of shape {frame.shape}."
            )

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = self._transform(crop_rgb).unsqueeze(0)
        tensor = tensor.to(torch.device(self.device))

        with torch.no_grad():
            feat = self._model(tensor)

        embedding = feat.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def match(
        self,
        query_embedding: np.ndarray,
        camera_id: int,
    ) -> Optional[int]:
        """Match a query embedding against the gallery for all other cameras.

        Cosine similarity is used (dot product of L2-normalised vectors).
        The same-camera gallery is excluded to enforce cross-camera matching.

        Args:
            query_embedding: numpy array of shape (512,), L2-normalised.
            camera_id: Source camera ID — excluded from gallery search.

        Returns:
            Matched track_id (int) if cosine similarity >= similarity_threshold,
            else None.
        """
        best_score: float = -1.0
        best_track_id: Optional[int] = None

        for gal_cam_id, tracks in self._gallery.items():
            if gal_cam_id == camera_id:
                continue
            for track_id, gal_embedding in tracks.items():
                score = float(np.dot(query_embedding, gal_embedding))
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

        if best_score >= self.similarity_threshold:
            return best_track_id
        return None

    def update_gallery(
        self,
        embedding: np.ndarray,
        track_id: int,
        camera_id: int,
    ) -> None:
        """Store an embedding in the in-memory gallery and persist to PostgreSQL.

        The in-memory gallery is updated immediately. DB persistence is
        attempted and failures are logged as warnings without raising.

        Args:
            embedding: numpy array of shape (512,), L2-normalised.
            track_id: Track ID this embedding belongs to.
            camera_id: Camera ID this embedding was captured from.
        """
        if camera_id not in self._gallery:
            self._gallery[camera_id] = {}
        if track_id in self._gallery[camera_id]:
            # Running average: blend old and new embedding, then re-normalise
            updated = (
                self.GALLERY_ALPHA * self._gallery[camera_id][track_id]
                + (1 - self.GALLERY_ALPHA) * embedding
            )
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated = updated / norm
            self._gallery[camera_id][track_id] = updated
        else:
            self._gallery[camera_id][track_id] = embedding

        # Persist to PostgreSQL
        try:
            from datetime import datetime as dt

            from argus.api.database import SessionLocal
            from argus.api.models import Embedding

            db = SessionLocal()
            try:
                entry = Embedding(
                    track_id=track_id,
                    camera_id=camera_id,
                    embedding=embedding.tolist(),
                    timestamp=dt.utcnow(),
                )
                db.add(entry)
                db.commit()
            except Exception as db_exc:
                db.rollback()
                logger.warning(
                    "DB persistence failed for embedding (track=%d cam=%d): %s",
                    track_id, camera_id, db_exc,
                )
            finally:
                db.close()
        except ImportError as imp_exc:
            logger.warning("DB module unavailable, skipping persistence: %s", imp_exc)

    def clear_gallery(self) -> None:
        """Clear the in-memory gallery. DB entries are not deleted."""
        self._gallery.clear()
        logger.info("In-memory gallery cleared.")
