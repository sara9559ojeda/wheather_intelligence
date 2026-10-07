"""Endpoints de predicciones y métricas de los modelos."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.db.models import ModelRun, Prediction, RetrainingRun
from backend.app.db.queries import get_location, load_observations_df
from backend.app.schemas import HorizonPredictionOut, ModelRunOut, RetrainingRunOut
from backend.app.services.artifacts import get_predictor

router = APIRouter(tags=["predictions"])


@router.get("/predictions/latest", response_model=list[HorizonPredictionOut])
def latest_predictions(
    db: Session = Depends(get_db), slug: str = Depends(get_location_slug)
) -> list[HorizonPredictionOut]:
    """Predicción en vivo a partir de las observaciones más recientes."""
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(503, "Modelos no entrenados (ejecuta backend.ml.modeling.run_training)")
    recent = load_observations_df(db, slug, last_n=72)
    try:
        preds = predictor.predict_from_observations(recent)
    except ValueError as exc:
        raise HTTPException(503, f"No hay observaciones recientes suficientes: {exc}") from exc
    return [HorizonPredictionOut(**p.as_dict()) for p in preds]


@router.get("/predictions/history", response_model=list[HorizonPredictionOut])
def prediction_history(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    horizon_hours: int = 6,
    limit: int = 200,
) -> list[HorizonPredictionOut]:
    """Predicciones persistidas (para el gráfico predicho-vs-real)."""
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    rows = db.execute(
        select(Prediction, ModelRun.model_type)
        .join(ModelRun, ModelRun.id == Prediction.model_run_id)
        .where(Prediction.location_id == loc.id, ModelRun.horizon_hours == horizon_hours)
        .order_by(Prediction.target_time.desc())
        .limit(limit)
    ).all()
    return [
        HorizonPredictionOut(
            horizon_hours=horizon_hours,
            base_time=p.base_time, target_time=p.target_time,
            predicted_temperature=float(p.predicted_temperature),
            lower_bound=float(p.lower_bound) if p.lower_bound is not None else None,
            upper_bound=float(p.upper_bound) if p.upper_bound is not None else None,
            actual_temperature=(
                float(p.actual_temperature) if p.actual_temperature is not None else None
            ),
            model_type=mt,
        )
        for p, mt in reversed(rows)
    ]


@router.get("/model/runs", response_model=list[ModelRunOut])
def model_runs(
    db: Session = Depends(get_db), active_only: bool = False
) -> list[ModelRunOut]:
    stmt = select(ModelRun).order_by(ModelRun.horizon_hours, ModelRun.rmse)
    if active_only:
        stmt = stmt.where(ModelRun.is_active.is_(True))
    return [ModelRunOut.model_validate(r) for r in db.scalars(stmt).all()]


@router.get("/model/performance", response_model=list[ModelRunOut])
def model_performance(db: Session = Depends(get_db)) -> list[ModelRunOut]:
    """Solo los campeones activos (para la tarjeta 'Model Performance' del dashboard)."""
    rows = db.scalars(
        select(ModelRun)
        .where(ModelRun.is_active.is_(True), ModelRun.model_type.notlike("baseline%"))
        .order_by(ModelRun.horizon_hours)
    ).all()
    return [ModelRunOut.model_validate(r) for r in rows]


@router.get("/model/retraining-history", response_model=list[RetrainingRunOut])
def retraining_history(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    limit: int = 20,
) -> list[RetrainingRunOut]:
    """Historial de intentos de reentrenamiento (fase 9): promovidos y descartados."""
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    rows = db.scalars(
        select(RetrainingRun)
        .where(RetrainingRun.location_id == loc.id)
        .order_by(RetrainingRun.triggered_at.desc())
        .limit(limit)
    ).all()
    return [RetrainingRunOut.model_validate(r) for r in rows]
