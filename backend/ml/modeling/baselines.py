"""Modelos de referencia (*baselines*) — fase 5.

Un modelo de ML solo aporta valor si **supera** a estas referencias triviales.

- **Persistencia:** "dentro de H horas hará la misma temperatura que ahora".
  Sorprendentemente difícil de batir a 1-3 h vista.
- **Climatología estacional:** "dentro de H horas hará la temperatura media
  histórica de ese mes y esa hora". Referencia a horizontes largos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceBaseline:
    name = "baseline_persistence"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["temperature_2m"].to_numpy(dtype="float64")


class SeasonalBaseline:
    name = "baseline_seasonal"

    def __init__(self) -> None:
        self._table: pd.Series | None = None

    def fit(self, df_train: pd.DataFrame) -> SeasonalBaseline:
        tmp = df_train.copy()
        tmp["month"] = tmp["timestamp"].dt.month
        tmp["hour"] = tmp["timestamp"].dt.hour
        self._table = tmp.groupby(["month", "hour"])["temperature_2m"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._table is None:
            raise RuntimeError("SeasonalBaseline sin ajustar")
        key = pd.MultiIndex.from_arrays(
            [df["timestamp"].dt.month, df["timestamp"].dt.hour], names=["month", "hour"]
        )
        return self._table.reindex(key).to_numpy(dtype="float64")
