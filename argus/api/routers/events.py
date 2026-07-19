from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from argus.api.auth import get_current_user
from argus.api.database import SessionLocal
from argus.api.models import Event
from argus.api.schemas import EventResponse

router = APIRouter(prefix="/api/v1/events", tags=["events"])


@router.get("", response_model=List[EventResponse])
async def list_events(
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    _: dict = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        events = (
            db.query(Event)
            .order_by(Event.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            EventResponse(
                id=e.id,
                description=e.description,
                event_metadata=e.event_metadata or {},
            )
            for e in events
        ]
    finally:
        db.close()
