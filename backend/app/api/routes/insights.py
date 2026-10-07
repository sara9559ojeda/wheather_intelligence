"""Endpoint de interpretación con IA (Claude) — fase 7.

    resumen estructurado (Python)  ->  Claude  ->  texto en secciones fijas

Con caché (``ai_cache_minutes``) y *fallback* (si no hay clave o la API falla,
devuelve el último análisis disponible o solo el resumen).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.services.claude import get_or_create_interpretation

router = APIRouter(tags=["insights"])


@router.get("/insights")
def insights(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    force: bool = Query(False, description="Ignora la caché y vuelve a llamar a Claude"),
) -> dict:
    try:
        result = get_or_create_interpretation(db, slug, force=force)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    if result["status"] == "ok":
        db.commit()
    return result
