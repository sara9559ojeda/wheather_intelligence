"""Dependencias comunes de FastAPI."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.session import SessionLocal


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_location_slug() -> str:
    """Slug de la ubicación configurada (el proyecto trabaja con una)."""
    return settings.wi_location_slug
