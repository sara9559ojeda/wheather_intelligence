"""Limpieza de observaciones (fase 3).

Filosofía: corregir **solo errores inequívocos** (de unidad, signo o sensor),
nunca recortar extremos meteorológicos reales. Cada corrección se cuenta y se
devuelve en un informe.

Para el dataset ERA5 (Open-Meteo) el número de correcciones es ~0 (ver
``docs/fase-2-data-understanding.md``); el módulo existe porque los datos en
tiempo real y el contraste de estación sí los necesitan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

# Límites duros: fuera de esto es físicamente imposible -> se recorta.
_HARD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "relative_humidity_2m": (0.0, 100.0),
    "cloud_cover": (0.0, 100.0),
    "cloud_cover_low": (0.0, 100.0),
    "cloud_cover_mid": (0.0, 100.0),
    "cloud_cover_high": (0.0, 100.0),
    "precipitation": (0.0, None),
    "rain": (0.0, None),
    "snowfall": (0.0, None),
    "wind_speed_10m": (0.0, None),
    "wind_gusts_10m": (0.0, None),
    "wind_speed_100m": (0.0, None),
    "shortwave_radiation": (0.0, None),
    "direct_radiation": (0.0, None),
    "diffuse_radiation": (0.0, None),
    "wind_direction_10m": (0.0, 360.0),
}


@dataclass
class CleaningReport:
    n_rows_in: int
    n_rows_out: int
    duplicates_removed: int = 0
    clipped: dict[str, int] = field(default_factory=dict)

    def log(self) -> None:
        log.info("Limpieza: %d -> %d filas (%d duplicados eliminados)",
                 self.n_rows_in, self.n_rows_out, self.duplicates_removed)
        for col, n in self.clipped.items():
            if n:
                log.info("  %s: %d valores recortados a límite físico", col, n)


def clean_observations(
    df: pd.DataFrame, ts_col: str = "timestamp"
) -> tuple[pd.DataFrame, CleaningReport]:
    n_in = len(df)
    df = df.sort_values(ts_col).reset_index(drop=True)

    dup = int(df[ts_col].duplicated().sum())
    df = df.drop_duplicates(subset=ts_col, keep="first").reset_index(drop=True)

    report = CleaningReport(n_rows_in=n_in, n_rows_out=len(df), duplicates_removed=dup)

    for col, (lo, hi) in _HARD_BOUNDS.items():
        if col not in df.columns:
            continue
        s = df[col]
        mask = pd.Series(False, index=s.index)
        if lo is not None:
            mask |= s < lo
        if hi is not None:
            mask |= s > hi
        n = int(mask.sum())
        report.clipped[col] = n
        if n:
            df[col] = s.clip(lower=lo, upper=hi)

    report.n_rows_out = len(df)
    return df, report
