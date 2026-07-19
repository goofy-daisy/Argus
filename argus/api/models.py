from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Camera(Base):
    """Registered camera source."""

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)      # "rgb" or "thermal"
    location = Column(String(255), nullable=False)

    tracks = relationship("Track", back_populates="camera")


class Track(Base):
    """Persistent multi-object track spanning one or more frames."""

    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    frame_start = Column(Integer, nullable=False)
    frame_end = Column(Integer, nullable=True)
    bbox_history = Column(JSONB, nullable=False, default=list)

    camera = relationship("Camera", back_populates="tracks")


class Alert(Base):
    """Threat alert generated from a track's composite threat score."""
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, nullable=True)
    type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String(10), default="medium", nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class Event(Base):
    """Action recognition and system event log entry."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(String(1000), nullable=False)
    event_metadata = Column("metadata", JSONB, nullable=False, default=dict)


class Embedding(Base):
    """Per-track appearance embedding stored for cross-camera re-identification."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    embedding = Column(JSONB, nullable=False)   # 512-dim float32 stored as JSON list
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class Zone(Base):
    """Spatial alert zone defined as a polygon in normalised frame coordinates."""

    __tablename__ = "zones"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    name = Column(String(255), nullable=False)
    polygon = Column(JSONB, nullable=False, default=list)  # [[x,y], ...] fractions 0-1
    alert_on_enter = Column(Boolean, default=True, nullable=False)
    alert_on_dwell = Column(Boolean, default=False, nullable=False)
    dwell_threshold_seconds = Column(Float, default=30.0, nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CrossCameraLink(Base):
    """Cross-camera identity link produced by Re-ID matching."""

    __tablename__ = "cross_camera_links"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    identity_id = Column(UUID(as_uuid=False), nullable=False)
    source_track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)
    source_camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    dest_track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)
    dest_camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    linked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
