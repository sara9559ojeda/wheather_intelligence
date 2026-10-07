"""Endpoints de salud y metadatos."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.core.config import settings
from backend.app.db.models import WeatherObservation
from backend.app.db.queries import get_location

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(select(1))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


@router.get("/meta")
def meta(db: Session = Depends(get_db), slug: str = Depends(get_location_slug)) -> dict:
    loc = get_location(db, slug)
    n_hist = n_realtime = 0
    last_obs = None
    if loc is not None:
        n_hist = db.scalar(
            select(func.count()).select_from(WeatherObservation).where(
                WeatherObservation.location_id == loc.id,
                WeatherObservation.source == "historical",
            )
        )
        n_realtime = db.scalar(
            select(func.count()).select_from(WeatherObservation).where(
                WeatherObservation.location_id == loc.id,
                WeatherObservation.source == "realtime",
            )
        )
        last_obs = db.scalar(
            select(func.max(WeatherObservation.observed_at)).where(
                WeatherObservation.location_id == loc.id
            )
        )
    return {
        "location": settings.wi_location_name,
        "latitude": settings.wi_latitude,
        "longitude": settings.wi_longitude,
        "timezone": settings.wi_timezone,
        "observations": {"historical": n_hist, "realtime": n_realtime},
        "latest_observation": last_obs.isoformat() if last_obs else None,
        "ingest_interval_minutes": settings.ingest_interval_minutes,
    }
