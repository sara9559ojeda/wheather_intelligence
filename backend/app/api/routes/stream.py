"""Server-Sent Events: notifica al dashboard cuando entra una observación nueva.

Patrón "SSE notifica, REST trae los datos": el evento solo dice *qué* cambió;
el frontend hace *refetch* del endpoint correspondiente.

Se mantiene mientras el frontend desplegado lo use; el canal preferente es el
WebSocket ``/api/ws`` (ver ``ws.py``). Ambos leen del mismo bus de eventos.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from backend.app.services.events import CLOSED, bus

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
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                if message is CLOSED:  # el bus se cayó: se corta y el cliente reconecta
                    break
                yield {"event": "update", "data": message}
        finally:
            bus.unsubscribe(queue)
            log.debug("Cliente SSE desconectado (%d activos)", bus.n_subscribers)

    return EventSourceResponse(event_generator())
