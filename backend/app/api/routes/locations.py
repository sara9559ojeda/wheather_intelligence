"""Catálogo de ciudades disponibles: ``GET /api/locations``."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.core.config import settings
from backend.app.services.locations import ready_locations

router = APIRouter(tags=["locations"])


@router.get("/locations")
def list_locations(db: Session = Depends(get_db)) -> list[dict]:
    """Ciudades con modelos entrenados (la predeterminada primero).

    El resto de endpoints aceptan ``?location=<slug>`` con cualquiera de ellas.
    """
    return [
        {
            "slug": loc.slug,
            "name": loc.name,
            "country": loc.country,
            "latitude": float(loc.latitude),
            "longitude": float(loc.longitude),
            "timezone": loc.timezone,
            "is_default": loc.slug == settings.wi_location_slug,
        }
        for loc in ready_locations(db)
    ]
