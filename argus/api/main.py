from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from argus.api.inference_pipeline import registry
from argus.api.routers import alerts, auth, cameras, events, heatmap, identities, tracks, zones
from argus.api.websocket_manager import manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Argus API starting up")
    yield
    logger.info("Argus API shutting down — stopping all pipelines")
    registry.stop_all()


app = FastAPI(
    title="Argus API",
    description="Multi-Camera Threat Intelligence System",
    version="0.9.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST routers ──────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(tracks.router)
app.include_router(alerts.router)
app.include_router(events.router)
app.include_router(zones.router)
app.include_router(heatmap.router)
app.include_router(identities.router)


# ── WebSocket: live video stream ──────────────────────────────────────────────
@app.websocket("/ws/stream/{camera_id}")
async def ws_stream(websocket: WebSocket, camera_id: int):
    await manager.connect_stream(websocket, camera_id)
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await manager.disconnect_stream(websocket, camera_id)


# ── WebSocket: alert channel ──────────────────────────────────────────────────
@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await manager.connect_alerts(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect_alerts(websocket)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "argus-api",
        "phase": 9,
        "version": "0.9.0",
        "active_pipelines": list(registry._pipelines.keys()),
        "alert_clients": manager.alert_client_count(),
    }
