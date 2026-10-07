"""Endpoints de clima actual e histórico."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.core.config import settings
from backend.app.db.models import WeatherObservation
from backend.app.db.queries import get_location
from backend.app.schemas import CurrentWeatherOut, ObservationOut

router = APIRouter(tags=["weather"])

_DELAY_THRESHOLD_MIN = 120


def _latest_observation(db: Session, location_id: int) -> WeatherObservation | None:
    return db.scalar(
        select(WeatherObservation)
        .where(WeatherObservation.location_id == location_id)
        .order_by(WeatherObservation.observed_at.desc())
        .limit(1)
    )


@router.get("/weather/current", response_model=CurrentWeatherOut)
def current_weather(
    db: Session = Depends(get_db), slug: str = Depends(get_location_slug)
) -> CurrentWeatherOut:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    obs = _latest_observation(db, loc.id)
    if obs is None:
        raise HTTPException(503, "Todavía no hay observaciones cargadas")

    age = max(0.0, (datetime.now(UTC) - obs.observed_at).total_seconds() / 60)
    return CurrentWeatherOut(
        location=loc.name,
        latitude=float(loc.latitude),
        longitude=float(loc.longitude),
        observation=ObservationOut.model_validate(obs),
        age_minutes=round(age, 1),
        delayed=age > _DELAY_THRESHOLD_MIN,
    )


@router.get("/weather/history", response_model=list[ObservationOut])
def history(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    hours: int = Query(168, ge=1, le=24 * 400, description="Ventana hacia atrás en horas"),
    source: str | None = Query(None, pattern="^(historical|realtime)$"),
) -> list[ObservationOut]:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    stmt = (
        select(WeatherObservation)
        .where(
            WeatherObservation.location_id == loc.id,
            WeatherObservation.observed_at >= cutoff,
        )
        .order_by(WeatherObservation.observed_at)
    )
    if source:
        stmt = stmt.where(WeatherObservation.source == source)
    rows = db.scalars(stmt).all()
    return [ObservationOut.model_validate(r) for r in rows]


@router.get("/config")
def public_config() -> dict:
    """Datos no sensibles que el frontend necesita al arrancar."""
    return {
        "location": settings.wi_location_name,
        "latitude": settings.wi_latitude,
        "longitude": settings.wi_longitude,
        "timezone": settings.wi_timezone,
    }
