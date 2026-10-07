"""Aplicación FastAPI de Weather Intelligence.

Arranque (lifespan):
  1. enlaza el event loop con el bus de SSE
  2. backfill de las últimas 72 h desde Open-Meteo (para que el dashboard no esté vacío)
  3. arranca el scheduler de ingesta en tiempo real

OpenAPI/Swagger en /docs, esquema en /openapi.json.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import (
    analytics,
    dashboard,
    health,
    insights,
    predictions,
    stream,
    weather,
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
    bus.bind_loop(asyncio.get_running_loop())

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

    asyncio.create_task(_startup_ingest())
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()


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
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Error no controlado en %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})


for r in (health, weather, analytics, predictions, dashboard, insights, stream):
    app.include_router(r.router, prefix="/api")


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"name": "Weather Intelligence API", "docs": "/docs", "health": "/api/health"}
