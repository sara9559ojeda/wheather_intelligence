"""Detección de anomalías con Isolation Forest sobre residuales (fase 4).

Corrección de mentor C2 (fase 1): aplicar Isolation Forest a la temperatura
cruda marcaría "todo el verano" como anómalo respecto a la media anual. Aquí
se trabaja sobre **residuales des-estacionalizados**:

    residual = valor observado − valor esperado para ese (mes, hora)

El "valor esperado" (climatología) se estima **solo con el tramo de train**, de
modo que "anómalo" significa *"raro respecto al comportamiento histórico
aprendido"* — igual que ocurrirá con un dato nuevo en tiempo real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)

ANOMALY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
)
RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.02


def fit_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Media de cada variable por (mes, hora). Índice = MultiIndex (month, hour)."""
    tmp = df.copy()
    tmp["month"] = tmp["timestamp"].dt.month
    tmp["hour"] = tmp["timestamp"].dt.hour
    clim = tmp.groupby(["month", "hour"])[list(ANOMALY_VARIABLES)].mean()
    return clim


def compute_residuals(df: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays(
        [df["timestamp"].dt.month, df["timestamp"].dt.hour], names=["month", "hour"]
    )
    out = {}
    for v in ANOMALY_VARIABLES:
        expected = climatology[v].reindex(key).to_numpy()
        out[f"{v}_resid"] = df[v].to_numpy(dtype="float64") - expected
    return pd.DataFrame(out, index=df.index)


@dataclass
class AnomalyModel:
    climatology: pd.DataFrame
    iforest: IsolationForest
    contamination: float
    resid_columns: list[str]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame con ``anomaly_score`` y ``is_anomaly`` por fila.

        ``anomaly_score``: cuanto **más negativo**, más anómalo (``score_samples``
        de Isolation Forest). Filas con residuales incompletos → NaN / False.
        """
        res = compute_residuals(df, self.climatology)
        valid = res.notna().all(axis=1).to_numpy()
        scores = np.full(len(df), np.nan)
        is_anom = np.zeros(len(df), dtype=bool)
        if valid.any():
            rv = res.to_numpy()[valid]
            scores[valid] = self.iforest.score_samples(rv)
            is_anom[valid] = self.iforest.predict(rv) == -1
        return pd.DataFrame(
            {"anomaly_score": scores, "is_anomaly": is_anom}, index=df.index
        )


def fit(df_train: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION) -> AnomalyModel:
    clim = fit_climatology(df_train)
    res = compute_residuals(df_train, clim).dropna()
    log.info("Isolation Forest: %d residuales de entrenamiento, contamination=%.3f",
             len(res), contamination)
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(res.to_numpy())
    return AnomalyModel(
        climatology=clim,
        iforest=iforest,
        contamination=contamination,
        resid_columns=list(res.columns),
    )
