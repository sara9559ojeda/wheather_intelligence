"""Pruebas del tiempo real y del modo serverless.

* Bus de eventos en memoria (pub/sub, publicación desde otro hilo, baja).
* WebSocket ``/api/ws``: recibe un evento publicado y el latido ``ping``.
* ``POST /api/tasks/ingest``: autenticación por token (la ingesta se simula).
* Lifespan con ``ENABLE_SCHEDULER`` / ``RUN_STARTUP_BACKFILL`` desactivados.
* Configuración serverless (URL de Neon, ``DB_POOL_MODE=null``).

Nada de esto necesita PostgreSQL ni red: ``TestClient`` sin context manager no
ejecuta el lifespan. Las pruebas marcadas ``redis`` necesitan un Redis en
``REDIS_TEST_URL`` (por defecto ``redis://localhost:6379/15``; ``docker compose
up -d redis``) y se omiten con ``-m 'not redis'``.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app import main as app_main
from backend.app.api.routes import ws as ws_route
from backend.app.core.config import Settings, settings
from backend.app.db.session import _engine_kwargs
from backend.app.main import app
from backend.app.scheduler import jobs
from backend.app.services.events import CLOSED, EventBus

REDIS_TEST_URL = os.environ.get("REDIS_TEST_URL", "redis://localhost:6379/15")
client = TestClient(app)


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Bus en memoria
# ---------------------------------------------------------------------------
def test_memory_bus_delivers_to_every_subscriber():
    async def scenario():
        bus = EventBus()
        assert bus.backend == "memory"
        q1, q2 = await bus.subscribe(), await bus.subscribe()
        assert bus.n_subscribers == 2
        bus.publish("observation", {"temperature_2m": 12.5})
        got = [await asyncio.wait_for(q.get(), 1) for q in (q1, q2)]
        bus.unsubscribe(q1)
        bus.unsubscribe(q2)
        return got, bus.n_subscribers

    got, remaining = asyncio.run(scenario())
    expected = '{"event": "observation", "data": {"temperature_2m": 12.5}}'
    assert got == [expected, expected]
    assert remaining == 0


def test_memory_bus_publish_from_another_thread():
    """El scheduler publica desde su hilo: debe llegar al loop del suscriptor."""
    async def scenario():
        bus = EventBus()
        q = await bus.subscribe()
        t = threading.Thread(target=bus.publish, args=("model_updated", {"status": "promoted"}))
        t.start()
        msg = await asyncio.wait_for(q.get(), 1)
        t.join()
        bus.unsubscribe(q)
        return msg

    assert '"model_updated"' in asyncio.run(scenario())


def test_memory_bus_unsubscribed_queue_gets_nothing():
    async def scenario():
        bus = EventBus()
        q = await bus.subscribe()
        bus.unsubscribe(q)
        bus.publish("observation", {})
        await asyncio.sleep(0.05)
        return q.empty()

    assert asyncio.run(scenario())


def test_publish_without_subscribers_is_noop():
    EventBus().publish("observation", {"x": 1})  # no lanza


# ---------------------------------------------------------------------------
# WebSocket /api/ws
# ---------------------------------------------------------------------------
@pytest.fixture
def memory_bus(monkeypatch):
    bus = EventBus()
    monkeypatch.setattr(ws_route, "bus", bus)
    monkeypatch.setattr(ws_route, "HEARTBEAT_SECONDS", 0.3)
    return bus


def test_websocket_receives_published_event_and_ping(memory_bus):
    with client.websocket_connect("/api/ws") as conn:
        hello = conn.receive_json()
        assert hello["type"] == "connected"
        assert memory_bus.n_subscribers == 1

        payload = {"observed_at": "2026-11-04T10:00:00+00:00", "temperature_2m": 14.2,
                   "is_anomaly": False}
        memory_bus.publish("observation", payload)
        assert conn.receive_json() == {"type": "observation", "data": payload}

        # sin más eventos, llega el latido
        assert conn.receive_json() == {"type": "ping"}

    # al desconectar el cliente se libera la suscripción (sin excepciones)
    assert _wait_until(lambda: memory_bus.n_subscribers == 0)


def test_websocket_closes_when_bus_subscription_dies(memory_bus):
    with client.websocket_connect("/api/ws") as conn:
        conn.receive_json()  # connected
        sub = next(iter(memory_bus._subs.values()))
        sub.loop.call_soon_threadsafe(sub.queue.put_nowait, CLOSED)
        msg = conn.receive()
        assert msg["type"] == "websocket.close"
        assert msg["code"] == 1011
    assert _wait_until(lambda: memory_bus.n_subscribers == 0)


def test_websocket_max_duration_closes_gracefully(memory_bus, monkeypatch):
    monkeypatch.setattr(settings, "ws_max_connection_seconds", 1)
    with client.websocket_connect("/api/ws") as conn:
        assert conn.receive_json()["data"]["max_connection_seconds"] == 1
        while True:
            msg = conn.receive()
            if msg["type"] == "websocket.close":
                break
        assert msg["code"] == 1012
    assert _wait_until(lambda: memory_bus.n_subscribers == 0)


# ---------------------------------------------------------------------------
# POST /api/tasks/ingest
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_ingest(monkeypatch):
    calls: list[int] = []

    def _run_ingest():
        calls.append(1)
        from datetime import UTC, datetime

        return {"observed_at": datetime(2026, 11, 4, 10, tzinfo=UTC),
                "comparison": {"is_anomaly": False}, "predictions": [{}, {}, {}]}

    monkeypatch.setattr(jobs, "run_ingest", _run_ingest)
    return calls


def test_ingest_disabled_when_token_empty(monkeypatch, fake_ingest):
    monkeypatch.setattr(settings, "ingest_token", "")
    r = client.post("/api/tasks/ingest", headers={"Authorization": "Bearer "})
    assert r.status_code == 503
    r = client.post("/api/tasks/ingest")
    assert r.status_code == 503
    assert fake_ingest == []


@pytest.mark.parametrize("headers", [
    {},                                              # sin cabecera
    {"Authorization": "Bearer otro-token"},          # token incorrecto
    {"Authorization": "Basic czNjcjN0LXRva2Vu"},     # esquema incorrecto
    {"Authorization": "Bearer s3cr3t-token-extra"},  # prefijo correcto, valor distinto
])
def test_ingest_rejects_missing_or_wrong_token(monkeypatch, fake_ingest, headers):
    monkeypatch.setattr(settings, "ingest_token", "s3cr3t-token")
    r = client.post("/api/tasks/ingest", headers=headers)
    assert r.status_code == 401
    assert fake_ingest == []


def test_ingest_runs_with_right_token(monkeypatch, fake_ingest):
    monkeypatch.setattr(settings, "ingest_token", "s3cr3t-token")
    r = client.post("/api/tasks/ingest", headers={"Authorization": "Bearer s3cr3t-token"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "observed_at": "2026-11-04T10:00:00+00:00",
                        "is_anomaly": False, "n_predictions": 3}
    assert fake_ingest == [1]


def test_ingest_failure_returns_502(monkeypatch):
    monkeypatch.setattr(settings, "ingest_token", "s3cr3t-token")

    def _boom():
        raise RuntimeError("Open-Meteo caído")

    monkeypatch.setattr(jobs, "run_ingest", _boom)
    r = client.post("/api/tasks/ingest", headers={"Authorization": "Bearer s3cr3t-token"})
    assert r.status_code == 502


def test_ingest_get_not_allowed():
    assert client.get("/api/tasks/ingest").status_code == 405


# ---------------------------------------------------------------------------
# Modo serverless: lifespan y configuración
# ---------------------------------------------------------------------------
def test_lifespan_skips_scheduler_and_backfill_when_disabled(monkeypatch):
    started: list[str] = []
    monkeypatch.setattr(settings, "enable_scheduler", False)
    monkeypatch.setattr(settings, "run_startup_backfill", False)
    monkeypatch.setattr(app_main, "start_scheduler", lambda: started.append("scheduler"))
    monkeypatch.setattr(app_main, "backfill_recent",
                        lambda *a, **k: started.append("backfill") or 0)
    monkeypatch.setattr(app_main, "ingest_job", lambda: started.append("ingest"))
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        time.sleep(0.1)
    assert started == []


def test_database_url_accepts_neon_style_scheme():
    url = "postgresql://user:pw@ep-x-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require"
    s = Settings(database_url=url, _env_file=None)
    assert s.database_url.startswith("postgresql+psycopg://user:pw@ep-x-pooler")
    s = Settings(database_url="postgres://u:p@h/db", _env_file=None)
    assert s.database_url == "postgresql+psycopg://u:p@h/db"
    s = Settings(database_url="postgresql+psycopg://u:p@h/db", _env_file=None)
    assert s.database_url == "postgresql+psycopg://u:p@h/db"


def test_engine_kwargs_null_pool_for_serverless():
    from sqlalchemy.pool import NullPool

    assert "poolclass" not in _engine_kwargs("queue")
    kw = _engine_kwargs("null")
    assert kw["poolclass"] is NullPool
    assert kw["pool_pre_ping"] is True


def test_serverless_defaults_keep_current_behaviour():
    s = Settings(_env_file=None)
    assert s.enable_scheduler is True
    assert s.run_startup_backfill is True
    assert s.db_pool_mode == "queue"
    assert s.redis_url == ""
    assert s.ingest_token == ""


# ---------------------------------------------------------------------------
# Redis (marcador `redis`)
# ---------------------------------------------------------------------------
@pytest.fixture
def redis_channel():
    import redis

    try:
        redis.Redis.from_url(REDIS_TEST_URL, socket_connect_timeout=2).ping()
    except redis.RedisError as exc:
        pytest.fail(f"Redis no disponible en {REDIS_TEST_URL}: {exc} "
                    "(docker compose up -d redis, o -m 'not redis')")
    return f"weather_intelligence:test:{uuid.uuid4().hex}"


@pytest.mark.redis
def test_redis_bus_fans_out_across_instances(redis_channel):
    """Dos buses = dos instancias: lo que publica una le llega a la otra."""
    async def scenario():
        subscriber_side = EventBus(REDIS_TEST_URL, channel=redis_channel)
        publisher_side = EventBus(REDIS_TEST_URL, channel=redis_channel)
        assert subscriber_side.backend == "redis"
        q = await subscriber_side.subscribe()
        # publish es síncrono (se llama desde hilos): igual que el scheduler
        await asyncio.to_thread(publisher_side.publish, "observation", {"temperature_2m": 9.1})
        msg = await asyncio.wait_for(q.get(), 5)
        subscriber_side.unsubscribe(q)
        publisher_side.close()
        return msg, subscriber_side.n_subscribers

    msg, remaining = asyncio.run(scenario())
    assert msg == '{"event": "observation", "data": {"temperature_2m": 9.1}}'
    assert remaining == 0


@pytest.mark.redis
def test_websocket_through_redis(redis_channel, monkeypatch):
    ws_bus = EventBus(REDIS_TEST_URL, channel=redis_channel)
    other_instance = EventBus(REDIS_TEST_URL, channel=redis_channel)
    monkeypatch.setattr(ws_route, "bus", ws_bus)
    with client.websocket_connect("/api/ws") as conn:
        assert conn.receive_json()["type"] == "connected"
        other_instance.publish("model_updated", {"status": "promoted",
                                                 "promoted_horizons": ["1"]})
        assert conn.receive_json() == {
            "type": "model_updated",
            "data": {"status": "promoted", "promoted_horizons": ["1"]},
        }
    assert _wait_until(lambda: ws_bus.n_subscribers == 0)
    other_instance.close()
