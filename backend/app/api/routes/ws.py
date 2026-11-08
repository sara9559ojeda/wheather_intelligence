"""WebSocket de tiempo real: ``GET /api/ws``.

Mismo patrón que el SSE de ``/api/stream`` ("el socket notifica, REST trae los
datos") y alimentado por el mismo bus de eventos (memoria o Redis pub/sub).

Autenticación: el navegador no puede enviar cabeceras en un WebSocket, así que
el cliente pide un ticket de 60 s con ``POST /api/auth/stream-ticket`` (con su
sesión) y conecta a ``/api/ws?ticket=<ticket>``. Sin ticket válido el
handshake se rechaza (HTTP 403 / cierre 1008). Hay que pedir un ticket nuevo
en cada (re)conexión.

Contrato de mensajes (servidor -> cliente, frames de texto JSON)::

    {"type": "connected", "data": {"heartbeat_seconds": 25,
                                   "max_connection_seconds": 0}}
        Primer mensaje. La suscripción al bus ya está activa: todo evento
        publicado a partir de aquí llega al cliente.

    {"type": "observation", "data": {"observed_at": "2026-11-04T10:00:00+00:00",
                                     "temperature_2m": 14.2, "is_anomaly": false}}
        Entró una observación nueva -> refetch de /api/dashboard (o de
        /api/weather/current, /api/predictions/latest...).

    {"type": "model_updated", "data": {"status": "promoted",
                                       "promoted_horizons": ["1", "24"]}}
        El reentrenamiento promocionó un modelo -> refetch de /api/model/*.

    {"type": "forecasts_verified", "data": {"location": "bogota", "n_verified": 5}}
        Se verificaron pronósticos emitidos (su hora objetivo ya pasó) -> refetch de
        /api/evaluation/* y /api/forecast/reliability de esa ciudad.

    {"type": "ping"}
        Latido si no ha habido otro mensaje en ``heartbeat_seconds``. No requiere
        respuesta; sirve para mantener viva la conexión y detectar cortes.

Cualquier otro ``type`` futuro se debe ignorar en el cliente. Los mensajes
cliente -> servidor se ignoran.

Cierre y reconexión: en Vercel la conexión se corta al llegar a la duración
máxima de la función (300 s en Hobby); con ``WS_MAX_CONNECTION_SECONDS`` el
servidor la cierra antes de forma ordenada (código 1012). Si el bus falla
(p. ej. se cae Redis) se cierra con 1011. En todos los casos el cliente debe
reconectar con *backoff* y hacer un refetch al reconectar, porque los eventos
emitidos mientras estaba desconectado no se reenvían.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from backend.app.core.config import settings
from backend.app.core.security import TOKEN_STREAM, AuthNotConfigured, InvalidToken, decode_token
from backend.app.services.events import CLOSED, bus

router = APIRouter(tags=["stream"])
log = logging.getLogger(__name__)

HEARTBEAT_SECONDS: float = 25

# Excepciones que solo significan "el cliente ya no está" (enviar a un socket cerrado)
_CLIENT_GONE = (WebSocketDisconnect, OSError, RuntimeError)


@router.websocket("/ws")
async def websocket_events(websocket: WebSocket) -> None:
    # Autenticación: ticket de 60 s (POST /api/auth/stream-ticket) en ?ticket=.
    # Se valida ANTES de aceptar: sin ticket válido el handshake se rechaza (HTTP 403).
    try:
        decode_token(websocket.query_params.get("ticket", ""), TOKEN_STREAM)
    except (InvalidToken, AuthNotConfigured):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    try:
        queue = await bus.subscribe()
    except Exception:  # noqa: BLE001 — sin bus no hay nada que enviar
        log.exception("No se pudo suscribir al bus de eventos")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    max_seconds = settings.ws_max_connection_seconds

    async def sender() -> None:
        await websocket.send_json({
            "type": "connected",
            "data": {"heartbeat_seconds": HEARTBEAT_SECONDS, "max_connection_seconds": max_seconds},
        })
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            if message is CLOSED:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                return
            payload = json.loads(message)
            await websocket.send_json({"type": payload["event"], "data": payload["data"]})

    async def receiver() -> None:
        # Solo para enterarse de la desconexión del cliente (los mensajes se ignoran).
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                return

    tasks = {asyncio.create_task(sender()), asyncio.create_task(receiver())}
    try:
        done, _ = await asyncio.wait(
            tasks,
            timeout=max_seconds or None,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:  # límite de duración alcanzado: cierre ordenado, el cliente reconecta
            await websocket.close(code=status.WS_1012_SERVICE_RESTART)
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, _CLIENT_GONE):
                log.error("Error en el WebSocket de eventos", exc_info=exc)
    except _CLIENT_GONE:
        pass  # el cliente se fue mientras cerrábamos
    finally:
        # primero lo síncrono: si este handler está siendo cancelado, los await
        # siguientes pueden no completarse, pero la baja del bus ya está hecha
        for task in tasks:
            task.cancel()
        bus.unsubscribe(queue)
        await asyncio.gather(*tasks, return_exceptions=True)
        log.debug("Cliente WebSocket desconectado (%d activos)", bus.n_subscribers)
