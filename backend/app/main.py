"""Aplicación FastAPI de Weather Intelligence.

Arranque (lifespan), cada paso desactivable para modo serverless (Vercel):
  1. backfill de las últimas 72 h desde Open-Meteo (para que el dashboard no esté
     vacío) + una ingesta de condiciones actuales   -> RUN_STARTUP_BACKFILL
  2. arranca el scheduler de ingesta en tiempo real -> ENABLE_SCHEDULER
En Vercel ambos van a false y la ingesta llega por POST /api/tasks/ingest.

Tiempo real: WebSocket /api/ws y SSE /api/stream, alimentados por el bus de
eventos (memoria, o Redis pub/sub si REDIS_URL está definido).

Acceso: login con email + contraseña (/api/auth/*, JWT). Salvo /api/health, /api/meta
y /api/auth/*, toda la API exige sesión; WebSocket y SSE usan un ticket de 60 s.

OpenAPI/Swagger en /docs, esquema en /openapi.json.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.deps import get_current_user
from backend.app.api.routes import (
    analytics,
    auth,
    dashboard,
    health,
    insights,
    predictions,
    stream,
    tasks,
    weather,
    ws,
)
from backend.app.core.config import settings
from backend.app.db.session import SessionLocal
from backend.app.scheduler.jobs import ingest_job, start_scheduler, stop_scheduler
from backend.app.services.events import bus
from backend.app.services.ingestion.open_meteo import OpenMeteoForecastClient
from backend.app.services.ingestion.realtime import backfill_recent

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Bus de eventos: %s | scheduler=%s | backfill inicial=%s",
             bus.backend, settings.enable_scheduler, settings.run_startup_backfill)
    if os.environ.get("VERCEL") and (settings.enable_scheduler or settings.run_startup_backfill):
        log.warning("Ejecutando en Vercel con ENABLE_SCHEDULER/RUN_STARTUP_BACKFILL activos: "
                    "en serverless deben ser false (la ingesta llega por /api/tasks/ingest)")

    async def _startup_ingest() -> None:
        client = OpenMeteoForecastClient(
            base_url=settings.open_meteo_forecast_url,
            latitude=settings.wi_latitude, longitude=settings.wi_longitude,
        )
        session = SessionLocal()
        try:
            n = await asyncio.to_thread(
                backfill_recent, session, client, settings.wi_location_slug, 3
            )
            session.commit()
            log.info("Backfill inicial: %d observaciones", n)
        except Exception:
            session.rollback()
            log.exception("Backfill inicial falló (se continúa)")
        finally:
            session.close()
        # una ingesta de "condiciones actuales" para tener el dato más fresco
        await asyncio.to_thread(ingest_job)

    startup_task = None
    if settings.run_startup_backfill:
        startup_task = asyncio.create_task(_startup_ingest())
    if settings.enable_scheduler:
        start_scheduler()
    try:
        yield
    finally:
        if settings.enable_scheduler:
            stop_scheduler()
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
        bus.close()


app = FastAPI(
    title="Weather Intelligence API",
    version="0.6.0",
    description="Minería de datos y ML meteorológico para Madrid-Barajas.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    # POST: login/registro/ticket desde el navegador. /api/tasks/* lo llama un servidor
    # (GitHub Actions) y no depende de CORS.
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Error no controlado en %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})


# Públicas: salud, login/registro. Con credenciales propias: tareas (INGEST_TOKEN) y
# tiempo real (ticket de 60 s). Todo lo demás exige una sesión de usuario (JWT).
for r in (health, auth, stream, ws, tasks):
    app.include_router(r.router, prefix="/api")
for r in (weather, analytics, predictions, dashboard, insights):
    app.include_router(r.router, prefix="/api", dependencies=[Depends(get_current_user)])


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"name": "Weather Intelligence API", "docs": "/docs", "health": "/api/health"}
