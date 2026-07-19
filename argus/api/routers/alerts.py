from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from argus.api.auth import get_current_user
from argus.api.database import SessionLocal
from argus.api.models import Alert
from argus.api.schemas import AlertAcknowledge, AlertResponse

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    camera_id: Optional[int] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    _: dict = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        q = db.query(Alert)
        # Filter by camera via track join when camera_id is provided
        if camera_id is not None:
            from argus.api.models import Track
            q = q.join(Track, Alert.track_id == Track.id).filter(
                Track.camera_id == camera_id
            )
        alerts = q.order_by(Alert.timestamp.desc()).offset(offset).limit(limit).all()
        return [
            AlertResponse(
                id=a.id,
                track_id=a.track_id,
                type=a.type,
                confidence=a.confidence,
                timestamp=a.timestamp,
            )
            for a in alerts
        ]
    finally:
        db.close()


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: int, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        a = db.query(Alert).filter(Alert.id == alert_id).first()
        if not a:
            raise HTTPException(status_code=404, detail="Alert not found")
        return AlertResponse(
            id=a.id,
            track_id=a.track_id,
            type=a.type,
            confidence=a.confidence,
            timestamp=a.timestamp,
        )
    finally:
        db.close()


@router.patch("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    payload: AlertAcknowledge,
    _: dict = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        a = db.query(Alert).filter(Alert.id == alert_id).first()
        if not a:
            raise HTTPException(status_code=404, detail="Alert not found")
        db.commit()
        return AlertResponse(
            id=a.id,
            track_id=a.track_id,
            type=a.type,
            confidence=a.confidence,
            acknowledged=payload.acknowledged,
            timestamp=a.timestamp,
        )
    finally:
        db.close()
