from typing import List

from fastapi import APIRouter, Depends

from argus.api.auth import get_current_user
from argus.api.database import SessionLocal
from argus.api.models import CrossCameraLink
from argus.api.schemas import CrossCameraLinkResponse

router = APIRouter(prefix="/api/v1/identities", tags=["identities"])


@router.get("/{identity_id}/links", response_model=List[CrossCameraLinkResponse])
async def get_identity_links(identity_id: str, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        links = (
            db.query(CrossCameraLink)
            .filter(CrossCameraLink.identity_id == identity_id)
            .order_by(CrossCameraLink.linked_at.desc())
            .all()
        )
        return links
    finally:
        db.close()


@router.get("", response_model=List[CrossCameraLinkResponse])
async def list_links(
    limit: int = 100, _: dict = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        links = (
            db.query(CrossCameraLink)
            .order_by(CrossCameraLink.linked_at.desc())
            .limit(limit)
            .all()
        )
        return links
    finally:
        db.close()
