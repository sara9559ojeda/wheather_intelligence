"""Crea o actualiza la fila de la ubicación del análisis en ``locations``.

Los datos vienen de ``backend/ml/config/location.toml`` (coordenadas pedidas)
y, si está disponible, del ``.meta.json`` del dataset ERA5 (elevación real de
la celda de rejilla).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Location
from backend.ml.config_loader import RAW_DIR, load_config

log = logging.getLogger(__name__)


def _grid_elevation_from_meta(slug: str) -> float | None:
    meta_path: Path = RAW_DIR / f"open_meteo_{slug}_hourly.meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("location", {}).get("grid_elevation_m")


def get_or_create_location(session: Session) -> Location:
    cfg = load_config()
    loc = session.scalar(select(Location).where(Location.slug == cfg.slug))
    elevation = _grid_elevation_from_meta(cfg.slug)

    if loc is None:
        loc = Location(
            name=cfg.name,
            slug=cfg.slug,
            latitude=cfg.latitude,
            longitude=cfg.longitude,
            timezone=cfg.timezone,
            country=cfg.country,
            elevation_m=elevation,
        )
        session.add(loc)
        session.flush()
        log.info("Ubicación creada: %s (id=%d)", loc.name, loc.id)
    else:
        loc.name = cfg.name
        loc.timezone = cfg.timezone
        loc.country = cfg.country
        if elevation is not None:
            loc.elevation_m = elevation
        log.info("Ubicación ya existente: %s (id=%d)", loc.name, loc.id)

    return loc
