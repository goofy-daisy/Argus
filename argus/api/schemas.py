from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ── Auth ─────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Camera ───────────────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name: str
    type: str = Field(..., pattern="^(rgb|thermal)$")
    location: str


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    location: Optional[str] = None


class CameraResponse(BaseModel):
    id: int
    name: str
    type: str
    location: str

    model_config = {"from_attributes": True}


# ── Track ────────────────────────────────────────────────────────────────────

class TrackResponse(BaseModel):
    id: int
    camera_id: int
    frame_start: int
    frame_end: Optional[int]
    label: Optional[str] = None
    anomaly_score: Optional[float] = None
    bbox_history: List[Any] = []

    model_config = {"from_attributes": True}


# ── Alert ────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    id: int
    track_id: int
    type: str
    confidence: float
    severity: Optional[str] = None
    acknowledged: bool = False
    timestamp: datetime

    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    acknowledged: bool = True


# ── Event ────────────────────────────────────────────────────────────────────

class EventResponse(BaseModel):
    id: int
    description: str
    event_metadata: dict = {}

    model_config = {"from_attributes": True}


# ── Zone ─────────────────────────────────────────────────────────────────────

class ZoneCreate(BaseModel):
    camera_id: int
    name: str
    polygon: List[List[float]] = Field(..., description="[[x, y], ...] in fractions 0-1")
    alert_on_enter: bool = True
    alert_on_dwell: bool = False
    dwell_threshold_seconds: float = 30.0
    active: bool = True


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    polygon: Optional[List[List[float]]] = None
    alert_on_enter: Optional[bool] = None
    alert_on_dwell: Optional[bool] = None
    dwell_threshold_seconds: Optional[float] = None
    active: Optional[bool] = None


class ZoneResponse(BaseModel):
    id: str
    camera_id: int
    name: str
    polygon: List[List[float]]
    alert_on_enter: bool
    alert_on_dwell: bool
    dwell_threshold_seconds: float
    active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Cross-Camera Link ─────────────────────────────────────────────────────────

class CrossCameraLinkResponse(BaseModel):
    id: str
    identity_id: str
    source_track_id: Optional[int]
    source_camera_id: int
    dest_track_id: Optional[int]
    dest_camera_id: int
    similarity_score: float
    linked_at: datetime

    model_config = {"from_attributes": True}


# ── Heatmap ──────────────────────────────────────────────────────────────────

class HeatmapResponse(BaseModel):
    camera_id: int
    grid: List[List[float]]
    grid_rows: int
    grid_cols: int
    timestamp: datetime


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    phase: int
    version: str
