"""Ubicaciones disponibles: las que tienen un modelo campeón entrenado y activo.

Una ciudad aparece en la API solo cuando su pipeline (``add_location``) ha
terminado; así el selector del frontend nunca ofrece una ciudad a medias.
"""

from __future__ import annotations

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Location, ModelRun


def ready_locations(session: Session) -> list[Location]:
    """Ubicaciones con modelos activos, la predeterminada primero y el resto por nombre."""
    has_model = exists().where(
        ModelRun.location_id == Location.id,
        ModelRun.is_active.is_(True),
        ModelRun.model_type.notlike("baseline%"),
    )
    rows = list(session.scalars(select(Location).where(has_model)).all())
    rows.sort(key=lambda loc: (loc.slug != settings.wi_location_slug, loc.name))
    return rows


def is_ready(session: Session, slug: str) -> bool:
    return any(loc.slug == slug for loc in ready_locations(session))
