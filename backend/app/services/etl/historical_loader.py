"""Validación y carga del dataset histórico en ``weather_observations``.

- La validación NO corrige nada: solo cuenta problemas y los registra. La
  limpieza/imputación es responsabilidad de la fase 3 (Data Preparation).
- La carga es **idempotente**: usa ``ON CONFLICT DO NOTHING`` sobre
  ``(location_id, observed_at, source)`` → se puede re-ejecutar sin duplicar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import SOURCE_HISTORICAL, WeatherObservation
from backend.ml.quality_rules import (
    PLAUSIBLE_RANGES,
    VALID_CATEGORICAL,
    count_out_of_range,
)

log = logging.getLogger(__name__)

# Columnas de medida de weather_observations que provienen de Open-Meteo.
MEASURE_COLUMNS: tuple[str, ...] = (
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "surface_pressure", "pressure_msl", "precipitation", "rain", "snowfall",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "wind_speed_100m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "et0_fao_evapotranspiration", "weather_code", "is_day",
)


@dataclass
class ValidationReport:
    n_rows: int
    duplicate_timestamps: int
    unsorted: bool
    nulls: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    out_of_domain: dict[str, int] = field(default_factory=dict)

    @property
    def total_issues(self) -> int:
        return (
            self.duplicate_timestamps
            + sum(self.out_of_range.values())
            + sum(self.out_of_domain.values())
        )

    def log(self) -> None:
        log.info("Validación: %d filas, %d duplicados, ordenado=%s",
                 self.n_rows, self.duplicate_timestamps, not self.unsorted)
        for col, n in self.out_of_range.items():
            if n:
                log.warning("  %s: %d valores fuera de rango plausible", col, n)
        for col, n in self.out_of_domain.items():
            if n:
                log.warning("  %s: %d valores fuera de dominio", col, n)
        nn = {c: n for c, n in self.nulls.items() if n}
        if nn:
            log.warning("  nulos: %s", nn)


def validate(df: pd.DataFrame) -> ValidationReport:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    rep = ValidationReport(
        n_rows=len(df),
        duplicate_timestamps=int(ts.duplicated().sum()),
        unsorted=not ts.is_monotonic_increasing,
    )
    for col in MEASURE_COLUMNS:
        if col not in df.columns:
            continue
        rep.nulls[col] = int(df[col].isna().sum())
        if col in PLAUSIBLE_RANGES:
            lo, hi = PLAUSIBLE_RANGES[col]
            rep.out_of_range[col] = count_out_of_range(df[col], lo, hi)
        if col in VALID_CATEGORICAL:
            observed = set(pd.to_numeric(df[col].dropna(), errors="coerce").unique())
            rep.out_of_domain[col] = len(observed - VALID_CATEGORICAL[col])
    return rep


def _to_records(df: pd.DataFrame, location_id: int, source: str) -> list[dict[str, Any]]:
    cols = ["timestamp", *[c for c in MEASURE_COLUMNS if c in df.columns]]
    clean = df[cols].copy()

    # tipos que la BD espera
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True).map(
        lambda t: t.to_pydatetime()
    )
    if "is_day" in clean:
        clean["is_day"] = clean["is_day"].map(lambda v: None if pd.isna(v) else bool(v))
    if "weather_code" in clean:
        clean["weather_code"] = clean["weather_code"].map(
            lambda v: None if pd.isna(v) else int(v)
        )

    # NaN -> None para el resto
    clean = clean.astype(object).where(pd.notna(clean), None)

    records = []
    for row in clean.to_dict("records"):
        rec = {
            "location_id": location_id,
            "observed_at": row.pop("timestamp"),
            "source": source,
        }
        rec.update(row)
        records.append(rec)
    return records


# PostgreSQL admite como máximo 65535 parámetros por sentencia.
_PG_MAX_PARAMS = 65535


def load_dataframe(
    session: Session,
    df: pd.DataFrame,
    *,
    location_id: int,
    source: str = SOURCE_HISTORICAL,
    chunk_size: int = 2_000,
) -> int:
    """Inserta ``df`` en weather_observations (idempotente).

    Devuelve el nº de filas procesadas (no el nº de filas *nuevas*: con
    ``ON CONFLICT DO NOTHING`` el ``rowcount`` no es fiable en todos los
    drivers; el llamador cuenta filas nuevas con COUNT(*) antes/después).
    """
    records = _to_records(df, location_id, source)
    if not records:
        return 0

    # Limita el lote para no superar el máximo de parámetros de PostgreSQL.
    n_cols = len(records[0])
    safe_chunk = max(1, min(chunk_size, _PG_MAX_PARAMS // n_cols))
    if safe_chunk < chunk_size:
        log.info("Lote ajustado a %d filas (%d columnas × %d ≤ %d params)",
                 safe_chunk, n_cols, safe_chunk, _PG_MAX_PARAMS)
    chunk_size = safe_chunk

    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        stmt = pg_insert(WeatherObservation).values(chunk).on_conflict_do_nothing(
            index_elements=["location_id", "observed_at", "source"]
        )
        session.execute(stmt)
        end = min(start + chunk_size, len(records))
        if end % 20_000 < chunk_size or end == len(records):
            log.info("  procesadas %d / %d filas", end, len(records))
    return len(records)
