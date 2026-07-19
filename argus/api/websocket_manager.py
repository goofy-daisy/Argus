from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for video streaming and alert broadcasting."""

    def __init__(self) -> None:
        # camera_id → set of connected WebSocket clients
        self._stream_clients: Dict[int, Set[WebSocket]] = {}
        # all clients subscribed to the global alert channel
        self._alert_clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ── Stream connections ────────────────────────────────────────────────────

    async def connect_stream(self, ws: WebSocket, camera_id: int) -> None:
        await ws.accept()
        async with self._lock:
            self._stream_clients.setdefault(camera_id, set()).add(ws)
        logger.info("WS stream connected: camera=%d  total=%d",
                    camera_id, len(self._stream_clients[camera_id]))

    async def disconnect_stream(self, ws: WebSocket, camera_id: int) -> None:
        async with self._lock:
            clients = self._stream_clients.get(camera_id, set())
            clients.discard(ws)
            if not clients:
                self._stream_clients.pop(camera_id, None)

    async def broadcast_bytes(self, camera_id: int, data: bytes) -> None:
        """Broadcast raw JPEG bytes to all stream subscribers for a camera."""
        clients = list(self._stream_clients.get(camera_id, set()))
        dead: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect_stream(ws, camera_id)

    # ── Alert connections ─────────────────────────────────────────────────────

    async def connect_alerts(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._alert_clients.add(ws)
        logger.info("WS alerts connected  total=%d", len(self._alert_clients))

    async def disconnect_alerts(self, ws: WebSocket) -> None:
        async with self._lock:
            self._alert_clients.discard(ws)

    async def broadcast_json(self, payload: dict) -> None:
        """Broadcast a JSON alert payload to all alert subscribers."""
        clients = list(self._alert_clients)
        dead: List[WebSocket] = []
        message = json.dumps(payload)
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect_alerts(ws)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stream_client_count(self, camera_id: int) -> int:
        return len(self._stream_clients.get(camera_id, set()))

    def alert_client_count(self) -> int:
        return len(self._alert_clients)


# Singleton used across the application
manager = ConnectionManager()
