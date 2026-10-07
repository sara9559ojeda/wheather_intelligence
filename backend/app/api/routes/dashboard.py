"""Endpoint agregado del dashboard + comparación histórica puntual."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.db.models import WeatherObservation
from backend.app.db.queries import get_location
from backend.app.schemas import DashboardOut, HistoricalComparisonOut
from backend.app.services.dashboard import build_dashboard
from backend.app.services.historical.comparison import compare

router = APIRouter(tags=["dashboard"])

_KEYS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
         "cloud_cover", "shortwave_radiation", "precipitation"]


@router.get("/dashboard", response_model=DashboardOut)
def dashboard(
    db: Session = Depends(get_db), slug: str = Depends(get_location_slug)
) -> DashboardOut:
    try:
        return build_dashboard(db, slug)
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/comparison/latest", response_model=HistoricalComparisonOut)
def latest_comparison(
    db: Session = Depends(get_db), slug: str = Depends(get_location_slug)
) -> HistoricalComparisonOut:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    obs = db.scalar(
        select(WeatherObservation)
        .where(WeatherObservation.location_id == loc.id)
        .order_by(WeatherObservation.observed_at.desc())
        .limit(1)
    )
    if obs is None:
        raise HTTPException(503, "Sin observaciones")
    values = {k: (float(getattr(obs, k)) if getattr(obs, k) is not None else None) for k in _KEYS}
    return HistoricalComparisonOut(**compare(db, slug, obs.observed_at, values))
