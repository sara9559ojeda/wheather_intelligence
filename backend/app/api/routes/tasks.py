"""Tareas disparadas desde fuera (modo serverless): ``POST /api/tasks/ingest``.

En Vercel no hay proceso de larga vida para APScheduler y el cron de Hobby
solo permite una ejecución al día, así que la ingesta de cada 30 min la lanza
un workflow programado de GitHub Actions (``.github/workflows/ingest.yml``)
contra este endpoint.

Seguridad: requiere ``Authorization: Bearer <INGEST_TOKEN>``. Si
``INGEST_TOKEN`` está vacío, el endpoint queda **deshabilitado** (503): nunca
abierto. El reentrenamiento NO se expone aquí (tarda ~5 min, más que la
duración máxima de una función).
"""

from __future__ import annotations

import asyncio
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.config import settings
from backend.app.scheduler import jobs

router = APIRouter(prefix="/tasks", tags=["tasks"])
log = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def require_ingest_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    expected = settings.ingest_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Endpoint deshabilitado: INGEST_TOKEN no configurado",
        )
    given = credentials.credentials if credentials is not None else ""
    if not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token no válido",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/ingest", dependencies=[Depends(require_ingest_token)])
async def trigger_ingest() -> dict:
    """Ejecuta una pasada de ingesta (la misma lógica que el job del scheduler:
    rellena huecos, procesa y publica el evento ``observation`` en el bus)."""
    try:
        summary = await asyncio.to_thread(jobs.run_ingest)
    except Exception:  # noqa: BLE001 — el detalle va al log, no al cliente
        log.exception("Fallo en la ingesta disparada por /api/tasks/ingest")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Fallo en la ingesta"
        ) from None
    if not summary:
        return {"status": "ok", "observed_at": None}
    comparison = summary.get("comparison") or {}
    return {
        "status": "ok",
        "observed_at": summary["observed_at"].isoformat(),
        "is_anomaly": comparison.get("is_anomaly"),
        "n_predictions": len(summary.get("predictions") or []),
    }
