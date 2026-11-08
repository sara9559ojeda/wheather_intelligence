"""Análisis contextual de la ciudad: normalidad histórica y regímenes meteorológicos.

Rutas síncronas (``def``): FastAPI las ejecuta en un hilo, así que los cálculos
con pandas no bloquean el bucle de eventos del servidor.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.schemas.analysis import NormalityOut, RegimesOut
from backend.app.services.analysis.normality import analyze_normality
from backend.app.services.analysis.regimes import analyze_regimes

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/normality", response_model=NormalityOut)
def normality(db: Session = Depends(get_db), slug: str = Depends(get_location_slug)) -> dict:
    """¿Qué tan normal es el tiempo actual respecto a observaciones comparables?"""
    try:
        return analyze_normality(db, slug)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/regimes", response_model=RegimesOut)
def regimes(db: Session = Depends(get_db), slug: str = Depends(get_location_slug)) -> dict:
    """Perfiles de los regímenes de la ciudad y régimen de la observación actual."""
    try:
        result = analyze_regimes(db, slug)
        db.commit()    # persiste el perfil la primera vez que se calcula
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return result
