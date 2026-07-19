from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from argus.api.auth import get_current_user
from argus.api.database import SessionLocal
from argus.api.models import Track
from argus.api.schemas import TrackResponse

router = APIRouter(prefix="/api/v1/tracks", tags=["tracks"])


@router.get("", response_model=List[TrackResponse])
async def list_tracks(
    camera_id: Optional[int] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    _: dict = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        q = db.query(Track)
        if camera_id is not None:
            q = q.filter(Track.camera_id == camera_id)
        tracks = q.order_by(Track.id.desc()).offset(offset).limit(limit).all()
        return [
            TrackResponse(
                id=t.id,
                camera_id=t.camera_id,
                frame_start=t.frame_start,
                frame_end=t.frame_end,
                bbox_history=t.bbox_history or [],
            )
            for t in tracks
        ]
    finally:
        db.close()


@router.get("/{track_id}", response_model=TrackResponse)
async def get_track(track_id: int, _: dict = Depends(get_current_user)):
    from fastapi import HTTPException
    db = SessionLocal()
    try:
        t = db.query(Track).filter(Track.id == track_id).first()
        if not t:
            raise HTTPException(status_code=404, detail="Track not found")
        return TrackResponse(
            id=t.id,
            camera_id=t.camera_id,
            frame_start=t.frame_start,
            frame_end=t.frame_end,
            bbox_history=t.bbox_history or [],
        )
    finally:
        db.close()
