from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# Heatmap grid resolution
_GRID_ROWS = 36
_GRID_COLS = 64


def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── Zone helpers (ray-casting point-in-polygon) ──────────────────────────────

def _point_in_polygon(px: float, py: float, polygon: List[List[float]]) -> bool:
    """Return True if (px, py) lies inside the polygon using ray casting."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_centre_norm(bbox: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h)


# ── Trajectory buffer ─────────────────────────────────────────────────────────

class _TrackBuffer:
    """Per-track rolling buffer for anomaly features and action clip frames."""

    SEQ_LEN = 30

    def __init__(self) -> None:
        self.feat_seq: Deque[List[float]] = deque(maxlen=self.SEQ_LEN)
        self.frame_buf: Deque[np.ndarray] = deque(maxlen=16)
        self.prev_cx: Optional[float] = None
        self.prev_cy: Optional[float] = None
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.first_seen: float = time.time()
        self.action_label: str = "normal"
        self.anomaly_score: float = 0.0

    def push_frame(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        vx = (cx - self.prev_cx) if self.prev_cx is not None else 0.0
        vy = (cy - self.prev_cy) if self.prev_cy is not None else 0.0
        bw, bh = (x2 - x1), (y2 - y1)
        aspect = bw / bh if bh > 0 else 1.0
        self.feat_seq.append([cx, cy, vx, vy, aspect])
        self.prev_cx = cx
        self.prev_cy = cy
        self.last_bbox = bbox
        self.frame_buf.append(frame.copy())

    def ready_for_anomaly(self) -> bool:
        return len(self.feat_seq) == self.SEQ_LEN

    def ready_for_action(self) -> bool:
        return len(self.frame_buf) == 16


# ── Composite threat score ────────────────────────────────────────────────────

def _composite_score(
    det_conf: float,
    action_label: str,
    anomaly_score: float,
    anomaly_threshold: float,
    weights: dict,
) -> float:
    action_risk = {
        "normal": 0.0,
        "loitering": 0.4,
        "running": 0.5,
        "falling": 0.7,
        "crowd_formation": 0.6,
    }.get(action_label, 0.3)
    anomaly_norm = min(anomaly_score / (anomaly_threshold + 1e-8), 1.0)
    score = (
        weights.get("detection", 0.4) * det_conf
        + weights.get("action", 0.35) * action_risk
        + weights.get("anomaly", 0.25) * anomaly_norm
    )
    return float(np.clip(score, 0.0, 1.0))


def _severity(score: float, high: float, medium: float) -> str:
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


# ── Gaussian heatmap kernel ───────────────────────────────────────────────────

def _gaussian_kernel(sigma: float = 1.0) -> np.ndarray:
    size = int(4 * sigma + 1) | 1  # odd
    k = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    for i in range(size):
        for j in range(size):
            k[i, j] = np.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * sigma ** 2))
    k /= k.sum() + 1e-8
    return k


_KERNEL = _gaussian_kernel(sigma=1.5)

def _persist_alert(db_session_factory, alert_data: dict) -> None:
    """Persist alert to database via raw SQL — bypasses ORM column mismatch."""
    try:
        import datetime
        from sqlalchemy import text
        db = db_session_factory()
        try:
            db.execute(text("""
                INSERT INTO alerts (track_id, type, confidence, severity, acknowledged, timestamp)
                VALUES (:track_id, :type, :confidence, :severity, :ack, :ts)
            """), {
                "track_id": alert_data.get("track_id"),
                "type":     alert_data["action"],
                "confidence": float(alert_data["threat_score"]),
                "severity": alert_data["severity"],
                "ack":      False,
                "ts":       datetime.datetime.utcnow(),
            })
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Alert DB persist failed: %s", exc)


def _update_heatmap(grid: np.ndarray, cx_norm: float, cy_norm: float) -> None:
    r = int(cy_norm * _GRID_ROWS)
    c = int(cx_norm * _GRID_COLS)
    r = max(0, min(_GRID_ROWS - 1, r))
    c = max(0, min(_GRID_COLS - 1, c))
    kh, kw = _KERNEL.shape
    r0, c0 = r - kh // 2, c - kw // 2
    for ki in range(kh):
        for kj in range(kw):
            gi, gj = r0 + ki, c0 + kj
            if 0 <= gi < _GRID_ROWS and 0 <= gj < _GRID_COLS:
                grid[gi, gj] += _KERNEL[ki, kj]


# ── Main pipeline class ───────────────────────────────────────────────────────

class InferencePipeline:
    """Integrates Detector → Tracker → ReIdentifier → ActionClassifier → AnomalyDetector.

    One pipeline instance per camera. Reads from camera.location (file path or
    RTSP URL), processes frames, broadcasts annotated JPEG bytes over WebSocket,
    and fires alert broadcasts for high-confidence threat scores.
    """

    def __init__(self, camera_id: int, camera_location: str) -> None:
        self.camera_id = camera_id
        self.camera_location = camera_location

        cfg = _load_config()
        self._cfg = cfg
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Per-track buffers: {track_id: _TrackBuffer}
        self._track_bufs: Dict[int, _TrackBuffer] = {}

        # Heatmap accumulator (decayed over time)
        self._heatmap: np.ndarray = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.float32)

        # Alert state to avoid duplicate firing
        self._alerted_tracks: Dict[int, float] = {}  # track_id → last alert time

        # Zone polygons for this camera (loaded from DB on start)
        self._zones: List[dict] = []

        self._detector = None
        self._tracker = None
        self._reider = None
        self._action = None
        self._anomaly = None

    # ── Public lifecycle ──────────────────────────────────────────────────────

    def start(self, ws_manager, alert_manager, db_session_factory) -> None:
        """Launch the async processing loop."""
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(
            self._run(ws_manager, alert_manager, db_session_factory)
        )

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def get_heatmap(self) -> List[List[float]]:
        """Return current heatmap grid as a nested list (rows × cols)."""
        normalised = self._heatmap.copy()
        mx = normalised.max()
        if mx > 0:
            normalised /= mx
        return normalised.tolist()

    def set_zones(self, zones: List[dict]) -> None:
        self._zones = zones

    # ── Internal async loop ───────────────────────────────────────────────────

    async def _run(self, ws_manager, alert_manager, db_session_factory) -> None:
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._load_models)
        except Exception as exc:
            logger.exception("Pipeline camera=%d: model load failed — %s", self.camera_id, exc)
            self._running = False
            return
        await self._load_zones_from_db(db_session_factory)

        _source_path = Path(self.camera_location)
        if _source_path.is_dir():
            _img_files = sorted(_source_path.glob("*.jpg"))
            if not _img_files:
                logger.error("Pipeline camera=%d: no jpg files in '%s'", self.camera_id, self.camera_location)
                self._running = False
                return
            cap = None
            _file_idx = 0
            logger.info("Pipeline started: camera=%d source='%s' (%d frames)",
                        self.camera_id, self.camera_location, len(_img_files))
        else:
            _img_files = []
            _file_idx = 0
            cap = cv2.VideoCapture(self.camera_location)
            if not cap.isOpened():
                logger.error("Pipeline camera=%d: cannot open '%s'", self.camera_id, self.camera_location)
                self._running = False
                return
            logger.info("Pipeline started: camera=%d source='%s'", self.camera_id, self.camera_location)

        frame_idx = 0
        try:
            while self._running:
                if _img_files:
                    frame = cv2.imread(str(_img_files[_file_idx]))
                    _file_idx = (_file_idx + 1) % len(_img_files)
                    if frame is None:
                        continue
                    ret = True
                else:
                    ret, frame = cap.read()
                if not ret:
                    if self.camera_location.startswith("rtsp"):
                        logger.warning("Camera %d: RTSP stream interrupted", self.camera_id)
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_idx += 1
                annotated, alerts = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_frame, frame, frame_idx
                )

                # Broadcast JPEG frame
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await ws_manager.broadcast_bytes(self.camera_id, buf.tobytes())

                # Broadcast alerts and persist to database
                for alert in alerts:
                    await alert_manager.broadcast_json(alert)
                    await asyncio.get_event_loop().run_in_executor(
                        None, _persist_alert, db_session_factory, alert
                    )

                # Gentle throttle — avoid spinning at GPU speed when no WS clients
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Pipeline camera=%d fatal error: %s", self.camera_id, exc)
        finally:
            if cap is not None:
                cap.release()
            self._running = False
            logger.info("Pipeline stopped: camera=%d", self.camera_id)

    def _load_models(self) -> None:
        cfg = self._cfg
        det_cfg = cfg["detection"]
        trk_cfg = cfg["tracking"]
        reid_cfg = cfg["reid"]
        act_cfg = cfg["action"]
        ano_cfg = cfg["anomaly"]

        from argus.action.action_classifier import ActionClassifier
        from argus.anomaly.anomaly_detector import AnomalyDetector
        from argus.detection.detector import Detector
        from argus.reid.reid import ReIdentifier
        from argus.tracking.tracker import Tracker

        self._detector = Detector(config=cfg)
        try:
            self._detector.load_model()
        except Exception as exc:
            logger.warning("Detector load failed (%s) — detections disabled", exc)
            self._detector = None

        self._tracker = Tracker(config=cfg)

        self._reider = ReIdentifier(
            model_path=str(_PROJECT_ROOT / reid_cfg["model_path"]),
            device=reid_cfg["device"],
            similarity_threshold=reid_cfg["similarity_threshold"],
            input_height=reid_cfg["input_height"],
            input_width=reid_cfg["input_width"],
            num_train_pids=reid_cfg["num_train_pids"],
        )
        try:
            self._reider.load_model()
        except Exception as exc:
            logger.warning("ReIdentifier load failed (%s) — Re-ID disabled", exc)
            self._reider = None

        self._action = ActionClassifier(
            model_path=str(_PROJECT_ROOT / act_cfg["model_path"]),
            device=act_cfg["device"],
            num_classes=act_cfg["num_classes"],
            clip_frames=act_cfg["clip_frames"],
            clip_size=act_cfg["clip_size"],
            video_mean=act_cfg["video_mean"],
            video_std=act_cfg["video_std"],
        )
        try:
            self._action.load_model()
        except Exception as exc:
            logger.warning("ActionClassifier load failed (%s) — action disabled", exc)
            self._action = None

        self._anomaly = AnomalyDetector(
            model_path=str(_PROJECT_ROOT / ano_cfg["model_path"]),
            device=ano_cfg["device"],
            threshold=ano_cfg["threshold"],
            sequence_length=ano_cfg["sequence_length"],
            feature_dim=ano_cfg["feature_dim"],
            hidden_size=ano_cfg["hidden_size"],
            num_layers=ano_cfg["num_layers"],
        )
        try:
            self._anomaly.load_model()
        except Exception as exc:
            logger.warning("AnomalyDetector load failed (%s) — anomaly disabled", exc)
            self._anomaly = None

        logger.info("Models loaded for camera=%d", self.camera_id)

    async def _load_zones_from_db(self, db_session_factory) -> None:
        try:
            from argus.api.models import Zone
            db = db_session_factory()
            try:
                zones = db.query(Zone).filter(
                    Zone.camera_id == self.camera_id, Zone.active == True
                ).all()
                self._zones = [
                    {"polygon": z.polygon, "name": z.name,
                     "alert_on_enter": z.alert_on_enter}
                    for z in zones
                ]
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Could not load zones for camera=%d: %s", self.camera_id, exc)

    def _process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Tuple[np.ndarray, List[dict]]:
        cfg = self._cfg
        alert_cfg = cfg["alerts"]
        h, w = frame.shape[:2]
        alerts: List[dict] = []
        annotated = frame.copy()

        # ── Detection ─────────────────────────────────────────────────────────
        if self._detector is not None:
            try:
                raw_dets = self._detector.detect(frame)
            except Exception:
                raw_dets = []
        else:
            raw_dets = []

        # ── Tracking ──────────────────────────────────────────────────────────
        if self._tracker is not None:
            try:
                tracks = self._tracker.update(raw_dets, frame)
            except Exception:
                tracks = []
        else:
            tracks = []

        # ── Per-track processing ───────────────────────────────────────────────
        active_ids = set()
        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Bounding box draw
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Track buffer
            if tid not in self._track_bufs:
                self._track_bufs[tid] = _TrackBuffer()
            buf = self._track_bufs[tid]
            buf.push_frame(frame, bbox, w, h)

            # Heatmap update
            cx_n, cy_n = _bbox_centre_norm(bbox, w, h)
            _update_heatmap(self._heatmap, cx_n, cy_n)

            # Decay heatmap every 30 frames
            if frame_idx % 30 == 0:
                self._heatmap *= 0.98

            # ── Anomaly detection ──────────────────────────────────────────────
            if self._anomaly is not None and buf.ready_for_anomaly():
                try:
                    seq = np.array(list(buf.feat_seq), dtype=np.float32)
                    buf.anomaly_score = self._anomaly.score(seq)
                except Exception:
                    pass

            # ── Action classification (every 8 frames) ─────────────────────────
            if self._action is not None and buf.ready_for_action() and frame_idx % 8 == 0:
                try:
                    clip_frames = list(buf.frame_buf)
                    clip = self._action.extract_clip(clip_frames, bbox)
                    label, _conf = self._action.classify(clip)
                    buf.action_label = label
                except Exception:
                    pass

            # ── Zone check ────────────────────────────────────────────────────
            in_zone = False
            zone_name = ""
            for zone in self._zones:
                if zone.get("alert_on_enter") and _point_in_polygon(cx_n, cy_n, zone["polygon"]):
                    in_zone = True
                    zone_name = zone.get("name", "zone")
                    break

            # ── Composite threat score ─────────────────────────────────────────
            det_conf = raw_dets[0][4] if raw_dets else 0.5
            threat = _composite_score(
                det_conf=det_conf,
                action_label=buf.action_label,
                anomaly_score=buf.anomaly_score,
                anomaly_threshold=self._anomaly.threshold if self._anomaly else 0.05,
                weights=alert_cfg["composite_weights"],
            )
            sev = _severity(threat, alert_cfg["high_threshold"], alert_cfg["medium_threshold"])

            # ── Overlay text ──────────────────────────────────────────────────
            label_text = f"ID:{tid} {buf.action_label} T:{threat:.2f}"
            cv2.putText(annotated, label_text, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # ── Alert emission (throttle: 1/track/10s) ─────────────────────────
            now = time.time()
            last = self._alerted_tracks.get(tid, 0.0)
            should_alert = (
                (threat >= alert_cfg["medium_threshold"] or in_zone)
                and (now - last) > 10.0
            )
            if should_alert:
                self._alerted_tracks[tid] = now
                alert_payload = {
                    "type": "alert",
                    "camera_id": self.camera_id,
                    "track_id": tid,
                    "threat_score": round(threat, 4),
                    "severity": sev,
                    "action": buf.action_label,
                    "anomaly_score": round(buf.anomaly_score, 4),
                    "in_zone": in_zone,
                    "zone_name": zone_name,
                    "timestamp": time.time(),
                }
                alerts.append(alert_payload)

                color = (0, 0, 255) if sev == "high" else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Clean up stale buffers
        stale = [tid for tid in self._track_bufs if tid not in active_ids]
        for tid in stale:
            del self._track_bufs[tid]

        # ── Frame timestamp overlay ───────────────────────────────────────────
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, f"CAM {self.camera_id}  {ts}", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated, alerts


# ── Pipeline registry ─────────────────────────────────────────────────────────

class PipelineRegistry:
    """Maps camera_id → running InferencePipeline instances."""

    def __init__(self) -> None:
        self._pipelines: Dict[int, InferencePipeline] = {}

    def start(self, camera_id: int, location: str, ws_manager, alert_manager, db_session_factory) -> None:
        if camera_id in self._pipelines:
            return
        pipe = InferencePipeline(camera_id, location)
        self._pipelines[camera_id] = pipe
        pipe.start(ws_manager, alert_manager, db_session_factory)

    def stop(self, camera_id: int) -> None:
        pipe = self._pipelines.pop(camera_id, None)
        if pipe:
            pipe.stop()

    def stop_all(self) -> None:
        for pipe in list(self._pipelines.values()):
            pipe.stop()
        self._pipelines.clear()

    def get_heatmap(self, camera_id: int) -> Optional[List[List[float]]]:
        pipe = self._pipelines.get(camera_id)
        return pipe.get_heatmap() if pipe else None

    def is_running(self, camera_id: int) -> bool:
        return camera_id in self._pipelines


registry = PipelineRegistry()
