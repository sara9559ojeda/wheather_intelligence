"""Pronóstico en vivo = modelo entrenado + intervalo calibrado.

Punto único para dashboard, ``/predictions/latest`` y la ingesta, de modo que
todos muestren (y registren) exactamente el mismo intervalo.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.db.queries import get_location, load_observations_df
from backend.app.services.artifacts import get_predictor
from backend.app.services.evaluation.uncertainty import apply_calibration
from backend.app.services.prediction.predictor import HorizonPrediction


def live_forecasts(session: Session, slug: str) -> list[HorizonPrediction]:
    """Pronósticos a todos los horizontes. [] si no hay modelos; ValueError si faltan datos."""
    predictor = get_predictor(slug)
    loc = get_location(session, slug)
    if predictor is None or loc is None:
        return []
    recent = load_observations_df(session, slug, last_n=72)
    # Descarta las filas de horas futuras (valores previstos guardados por adelantado).
    recent = recent[recent["timestamp"] <= pd.Timestamp.now(tz="UTC").floor("h")]
    preds = predictor.predict_from_observations(recent)
    return apply_calibration(session, loc.id, preds)
