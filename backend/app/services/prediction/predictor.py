"""Inferencia de temperatura a varios horizontes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from backend.app.services.features.builder import HORIZONS_HOURS, build_feature_frame

log = logging.getLogger(__name__)


@dataclass
class HorizonPrediction:
    horizon_hours: int
    base_time: pd.Timestamp
    target_time: pd.Timestamp
    predicted_temperature: float
    lower_bound: float
    upper_bound: float
    model_type: str

    def as_dict(self) -> dict:
        return {
            "horizon_hours": self.horizon_hours,
            "base_time": self.base_time.isoformat(),
            "target_time": self.target_time.isoformat(),
            "predicted_temperature": self.predicted_temperature,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "model_type": self.model_type,
        }


class TemperaturePredictor:
    def __init__(self, artifacts_dir: Path, slug: str) -> None:
        self.slug = slug
        self.models: dict[int, dict] = {}
        for h in HORIZONS_HOURS:
            path = artifacts_dir / f"model_temp_h{h}.joblib"
            if path.exists():
                self.models[h] = joblib.load(path)
        if not self.models:
            raise FileNotFoundError(
                f"No hay artefactos model_temp_h*.joblib en {artifacts_dir}. "
                "Ejecuta antes: python -m backend.ml.modeling.run_training"
            )
        log.info("Predictor cargado: horizontes %s", sorted(self.models))

    # ------------------------------------------------------------------ #
    def predict_from_feature_row(self, feature_row: pd.DataFrame) -> list[HorizonPrediction]:
        """``feature_row``: DataFrame de UNA fila con ``timestamp`` + las features."""
        base_time = pd.to_datetime(feature_row["timestamp"].iloc[0], utc=True)
        out: list[HorizonPrediction] = []
        for h in sorted(self.models):
            m = self.models[h]
            x = feature_row.loc[:, m["feature_columns"]]
            p = float(m["pipeline"].predict(x)[0])
            out.append(HorizonPrediction(
                horizon_hours=h,
                base_time=base_time,
                target_time=base_time + pd.Timedelta(hours=h),
                predicted_temperature=round(p, 2),
                lower_bound=round(p + m["residual_p05"], 2),
                upper_bound=round(p + m["residual_p95"], 2),
                model_type=m["model_type"],
            ))
        return out

    def predict_from_observations(self, observations: pd.DataFrame) -> list[HorizonPrediction]:
        """``observations``: histórico horario reciente (≥ 25 h) con columnas crudas.

        Se construyen las features y se predice a partir de la **última** fila.
        """
        if len(observations) < 25:
            raise ValueError("Se necesitan al menos 25 horas de observaciones recientes.")
        feats = build_feature_frame(observations, drop_incomplete=False)
        last = feats.iloc[[-1]].reset_index(drop=True)
        needed = self.models[next(iter(self.models))]["feature_columns"]
        missing = last[needed].isna().any(axis=1).iloc[0]
        if missing:
            raise ValueError(
                "La última fila tiene features incompletas: faltan horas recientes contiguas."
            )
        return self.predict_from_feature_row(last)
