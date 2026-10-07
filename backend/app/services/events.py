"""Bus de eventos en proceso para Server-Sent Events (SSE).

El servicio de ingesta publica un evento cuando entra una observación nueva;
cada cliente SSE tiene su propia cola y recibe los eventos posteriores a su
conexión. Todo en memoria: suficiente para una sola instancia del backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        """Puede llamarse desde un hilo (el scheduler) o desde el event loop."""
        message = json.dumps({"event": event, "data": data}, default=str)
        if self._loop is None:
            return
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, message)
            except asyncio.QueueFull:
                log.warning("Cola SSE llena; se descarta un evento para un cliente")

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)


bus = EventBus()
