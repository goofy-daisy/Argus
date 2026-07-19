from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from argus.api.auth import get_current_user
from argus.api.database import SessionLocal
from argus.api.models import Zone
from argus.api.schemas import ZoneCreate, ZoneResponse, ZoneUpdate

router = APIRouter(prefix="/api/v1/zones", tags=["zones"])


@router.get("", response_model=List[ZoneResponse])
async def list_zones(
    camera_id: int | None = None, _: dict = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        q = db.query(Zone)
        if camera_id is not None:
            q = q.filter(Zone.camera_id == camera_id)
        return q.all()
    finally:
        db.close()


@router.post("", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
async def create_zone(payload: ZoneCreate, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        zone = Zone(**payload.model_dump())
        db.add(zone)
        db.commit()
        db.refresh(zone)
        return zone
    finally:
        db.close()


@router.get("/{zone_id}", response_model=ZoneResponse)
async def get_zone(zone_id: str, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        z = db.query(Zone).filter(Zone.id == zone_id).first()
        if not z:
            raise HTTPException(status_code=404, detail="Zone not found")
        return z
    finally:
        db.close()


@router.patch("/{zone_id}", response_model=ZoneResponse)
async def update_zone(
    zone_id: str, payload: ZoneUpdate, _: dict = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        z = db.query(Zone).filter(Zone.id == zone_id).first()
        if not z:
            raise HTTPException(status_code=404, detail="Zone not found")
        for field, value in payload.model_dump(exclude_none=True).items():
            setattr(z, field, value)
        db.commit()
        db.refresh(z)
        return z
    finally:
        db.close()


@router.delete("/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_zone(zone_id: str, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        z = db.query(Zone).filter(Zone.id == zone_id).first()
        if not z:
            raise HTTPException(status_code=404, detail="Zone not found")
        db.delete(z)
        db.commit()
    finally:
        db.close()
