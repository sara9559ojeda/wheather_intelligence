"""Server-Sent Events: notifica al dashboard cuando entra una observación nueva.

Patrón "SSE notifica, REST trae los datos": el evento solo dice *qué* cambió;
el frontend hace *refetch* del endpoint correspondiente.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from backend.app.services.events import bus

router = APIRouter(tags=["stream"])
log = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 20


@router.get("/stream")
async def stream(request: Request) -> EventSourceResponse:
    queue = await bus.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
                    yield {"event": "update", "data": message}
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            bus.unsubscribe(queue)
            log.debug("Cliente SSE desconectado (%d activos)", bus.n_subscribers)

    return EventSourceResponse(event_generator())
