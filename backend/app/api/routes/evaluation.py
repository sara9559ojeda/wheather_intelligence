"""Evaluación transparente de los pronósticos y su incertidumbre.

- ``GET /api/evaluation/summary``: métricas de los pronósticos EMITIDOS y verificados.
- ``GET /api/evaluation/pairs``: pares predicho/observado para el gráfico.
- ``GET /api/forecast/reliability``: pronóstico vigente con su intervalo y su confiabilidad.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.db.queries import get_location
from backend.app.schemas.analysis import EvaluationOut, ForecastReliabilityOut, PairsOut
from backend.app.services.evaluation.forecast_log import evaluation_summary, verified_pairs
from backend.app.services.evaluation.uncertainty import reliability_report
from backend.app.services.prediction.live import live_forecasts

router = APIRouter(tags=["evaluation"])
log = logging.getLogger(__name__)


def _location(db: Session, slug: str):
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    return loc


@router.get("/evaluation/summary", response_model=EvaluationOut)
def summary(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    days: int = Query(30, ge=1, le=365, description="Ventana de evaluación en días"),
) -> dict:
    return evaluation_summary(db, _location(db, slug), days)


@router.get("/evaluation/pairs", response_model=PairsOut)
def pairs(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    days: int = Query(30, ge=1, le=365),
    horizon_hours: int = Query(6, ge=1, le=72),
    limit: int = Query(500, ge=1, le=2000),
) -> dict:
    return verified_pairs(db, _location(db, slug), days, horizon_hours, limit)


@router.get("/forecast/reliability", response_model=ForecastReliabilityOut)
def reliability(db: Session = Depends(get_db), slug: str = Depends(get_location_slug)) -> dict:
    loc = _location(db, slug)
    try:
        preds = live_forecasts(db, slug)
    except ValueError as exc:
        log.info("Sin pronóstico para la confiabilidad de %s: %s", slug, exc)
        preds = []
    return reliability_report(db, loc, preds)
