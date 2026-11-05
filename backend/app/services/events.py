"""Bus de eventos de tiempo real (alimenta WebSocket ``/api/ws`` y SSE ``/api/stream``).

El servicio de ingesta y los jobs publican un evento cuando cambia algo
(``observation``, ``model_updated``); cada cliente conectado tiene su propia
cola y recibe los eventos posteriores a su conexión.

Dos backends, elegidos por configuración:

* **Memoria** (``REDIS_URL`` vacío): todo en proceso. Suficiente para una sola
  instancia (desarrollo local, tests).
* **Redis pub/sub** (``REDIS_URL`` definido): ``publish`` hace ``PUBLISH`` en
  el canal y cada suscriptor abre su propia suscripción (cliente
  ``redis.asyncio``). Necesario en Vercel Functions, donde cada conexión
  WebSocket queda fijada a una instancia distinta y el evento puede generarse
  en otra (p. ej. la invocación de ``POST /api/tasks/ingest``).

Formato del mensaje en la cola / en el canal (texto JSON)::

    {"event": "<nombre>", "data": {...}}

``publish`` es **síncrono** y puede llamarse desde un hilo (scheduler,
``asyncio.to_thread``) o desde el event loop; en modo Redis usa el cliente
síncrono para no depender de ningún loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import settings

log = logging.getLogger(__name__)

#: Canal de Redis donde se publican los eventos.
CHANNEL = "weather_intelligence:events"
#: Tamaño máximo de la cola de cada cliente (si se llena, se descartan eventos).
QUEUE_MAXSIZE = 100

#: Elemento que se mete en la cola cuando la suscripción muere (p. ej. se cae
#: Redis): el consumidor debe cerrar la conexión para que el cliente reconecte.
CLOSED: None = None

Message = str | None


@dataclass(eq=False)
class _Subscription:
    queue: asyncio.Queue[Message]
    loop: asyncio.AbstractEventLoop
    reader: asyncio.Task[None] | None = field(default=None)


class EventBus:
    """Pub/sub con backend en memoria o en Redis (según ``redis_url``)."""

    def __init__(self, redis_url: str = "", channel: str = CHANNEL) -> None:
        self._redis_url = redis_url.strip()
        self._channel = channel
        self._subs: dict[asyncio.Queue[Message], _Subscription] = {}
        self._publisher: Any = None          # redis.Redis (síncrono), perezoso
        self._publisher_lock = threading.Lock()

    # ------------------------------------------------------------------ info
    @property
    def backend(self) -> str:
        return "redis" if self._redis_url else "memory"

    @property
    def n_subscribers(self) -> int:
        return len(self._subs)

    # ------------------------------------------------------------- suscribir
    async def subscribe(self) -> asyncio.Queue[Message]:
        """Crea la cola del cliente. En modo Redis, la suscripción ya está activa al volver."""
        sub = _Subscription(asyncio.Queue(maxsize=QUEUE_MAXSIZE), asyncio.get_running_loop())
        if self._redis_url:
            sub.reader = await self._start_redis_reader(sub.queue)
        self._subs[sub.queue] = sub
        return sub.queue

    def unsubscribe(self, queue: asyncio.Queue[Message]) -> None:
        """Da de baja la cola. Síncrono a propósito: se llama en bloques ``finally``
        de handlers que pueden estar siendo cancelados (cliente desconectado), donde
        un ``await`` podría no llegar a ejecutarse. El lector de Redis cierra sus
        conexiones él mismo al recibir la cancelación."""
        sub = self._subs.pop(queue, None)
        if sub is not None and sub.reader is not None and not sub.reader.done():
            sub.reader.cancel()

    async def _start_redis_reader(self, queue: asyncio.Queue[Message]) -> asyncio.Task[None]:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            health_check_interval=30,
        )
        pubsub = client.pubsub()
        try:
            await pubsub.subscribe(self._channel)
            # esperar la confirmación del SUBSCRIBE: así no se pierde un evento
            # publicado justo después de conectar
            confirmed = await pubsub.get_message(timeout=5)
            if not confirmed or confirmed.get("type") != "subscribe":
                raise ConnectionError("Redis no confirmó la suscripción al canal")
        except BaseException:
            await pubsub.aclose()
            await client.aclose()
            raise

        async def _reader() -> None:
            try:
                async for message in pubsub.listen():
                    if message.get("type") == "message":
                        _put(queue, message["data"])
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — se notifica al consumidor para que cierre
                log.exception("Suscripción a Redis caída")
                _put(queue, CLOSED, force=True)
            finally:
                with contextlib.suppress(Exception):
                    await pubsub.aclose()
                with contextlib.suppress(Exception):
                    await client.aclose()

        return asyncio.create_task(_reader(), name="redis-subscriber")

    # -------------------------------------------------------------- publicar
    def publish(self, event: str, data: dict[str, Any]) -> None:
        """Publica un evento. Nunca lanza: un fallo del bus no debe tumbar la ingesta."""
        message = json.dumps({"event": event, "data": data}, default=str)
        if self._redis_url:
            try:
                self._get_publisher().publish(self._channel, message)
            except Exception:  # noqa: BLE001
                log.exception("No se pudo publicar el evento '%s' en Redis", event)
            return
        for sub in list(self._subs.values()):
            try:
                sub.loop.call_soon_threadsafe(_put, sub.queue, message)
            except RuntimeError:  # loop cerrado: el cliente ya no existe
                self._subs.pop(sub.queue, None)

    def _get_publisher(self) -> Any:
        with self._publisher_lock:
            if self._publisher is None:
                import redis

                self._publisher = redis.Redis.from_url(
                    self._redis_url,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    health_check_interval=30,
                )
            return self._publisher

    def close(self) -> None:
        """Libera el cliente de publicación (al apagar la app)."""
        with self._publisher_lock:
            if self._publisher is not None:
                with contextlib.suppress(Exception):
                    self._publisher.close()
                self._publisher = None


def _put(queue: asyncio.Queue[Message], message: Message, *, force: bool = False) -> None:
    """Encola sin bloquear (se ejecuta en el loop del consumidor)."""
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        if not force:
            log.warning("Cola de eventos llena; se descarta un evento para un cliente")
            return
        # la señal de cierre tiene prioridad: se hace sitio descartando el más antiguo
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        queue.put_nowait(message)


bus = EventBus(redis_url=settings.redis_url)
