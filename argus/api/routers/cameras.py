from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from argus.api.auth import get_current_user
from argus.api.database import SessionLocal
from argus.api.inference_pipeline import registry
from argus.api.models import Camera
from argus.api.schemas import CameraCreate, CameraResponse, CameraUpdate
from argus.api.websocket_manager import manager

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("", response_model=List[CameraResponse])
async def list_cameras(_: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        return db.query(Camera).all()
    finally:
        db.close()


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(payload: CameraCreate, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        cam = Camera(name=payload.name, type=payload.type, location=payload.location)
        db.add(cam)
        db.commit()
        db.refresh(cam)
        return cam
    finally:
        db.close()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: int, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        return cam
    finally:
        db.close()


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int, payload: CameraUpdate, _: dict = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        for field, value in payload.model_dump(exclude_none=True).items():
            setattr(cam, field, value)
        db.commit()
        db.refresh(cam)
        return cam
    finally:
        db.close()


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_id: int, _: dict = Depends(get_current_user)):
    registry.stop(camera_id)
    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        db.delete(cam)
        db.commit()
    finally:
        db.close()


@router.post("/{camera_id}/start", status_code=status.HTTP_200_OK)
async def start_pipeline(camera_id: int, _: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        if registry.is_running(camera_id):
            return {"status": "already_running", "camera_id": camera_id}
        registry.start(camera_id, cam.location, manager, manager, SessionLocal)
        return {"status": "started", "camera_id": camera_id}
    finally:
        db.close()


@router.post("/{camera_id}/stop", status_code=status.HTTP_200_OK)
async def stop_pipeline(camera_id: int, _: dict = Depends(get_current_user)):
    registry.stop(camera_id)
    return {"status": "stopped", "camera_id": camera_id}


@router.get("/{camera_id}/status")
async def pipeline_status(camera_id: int, _: dict = Depends(get_current_user)):
    return {
        "camera_id": camera_id,
        "running": registry.is_running(camera_id),
        "stream_clients": manager.stream_client_count(camera_id),
    }
