"""Fixtures compartidas por las pruebas de la fase 11 (inteligencia contextual).

``synthetic_city`` crea una ciudad ficticia (zona horaria UTC, histórico sintético
y reproducible) y la borra al terminar, sin tocar los datos reales.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, insert

from backend.app.db.models import ForecastLog, Location, WeatherObservation
from backend.app.db.session import SessionLocal


def make_city_rows(
    start: str, end: str, *, hours: tuple[int, ...] | None = None, seed: int = 7,
    base: float = 15.0, amplitude: float = 8.0, noise: float = 3.0,
) -> pd.DataFrame:
    """Serie horaria sintética: ciclo anual + ruido gaussiano (determinista)."""
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    if hours is not None:
        idx = idx[idx.hour.isin(hours)]
    rng = np.random.default_rng(seed)
    doy = idx.dayofyear.to_numpy()
    season = amplitude * np.sin(2 * np.pi * (doy - 110) / 365.25)
    temp = base + season + rng.normal(0, noise, len(idx))
    return pd.DataFrame({
        "observed_at": idx,
        "temperature_2m": np.round(temp, 2),
        "relative_humidity_2m": np.round(
            np.clip(70 - 1.5 * (temp - base) + rng.normal(0, 8, len(idx)), 5, 100), 2),
        "wind_speed_10m": np.round(np.abs(rng.normal(8, 4, len(idx))), 2),
        "precipitation": np.round(
            np.where(rng.random(len(idx)) < 0.08, rng.exponential(1.2, len(idx)), 0.0), 1),
        "surface_pressure": np.round(rng.normal(1012, 6, len(idx)), 2),
    })


def insert_observations(session, location_id: int, df: pd.DataFrame, source: str = "historical",
                        ingested_at: datetime | None = None) -> None:
    rows = []
    for rec in df.to_dict("records"):
        row = {"location_id": location_id, "source": source,
               "observed_at": rec["observed_at"].to_pydatetime(),
               **{k: float(v) for k, v in rec.items() if k != "observed_at"}}
        if ingested_at is not None:
            row["ingested_at"] = ingested_at
        rows.append(row)
    for i in range(0, len(rows), 5000):
        session.execute(insert(WeatherObservation), rows[i:i + 5000])
    session.flush()


@contextmanager
def _city(df: pd.DataFrame | None, tz: str = "UTC"):
    slug = f"test_{uuid.uuid4().hex[:10]}"
    with SessionLocal() as s:
        loc = Location(name=f"Ciudad {slug}", slug=slug, latitude=1.0, longitude=1.0,
                       timezone=tz, country="CO")
        s.add(loc)
        s.flush()
        if df is not None:
            insert_observations(s, loc.id, df)
        s.commit()
        loc_id = loc.id
    try:
        yield slug, loc_id
    finally:
        with SessionLocal() as s:
            s.execute(delete(ForecastLog).where(ForecastLog.location_id == loc_id))
            s.execute(delete(WeatherObservation).where(WeatherObservation.location_id == loc_id))
            s.execute(delete(Location).where(Location.id == loc_id))
            s.commit()


@pytest.fixture(scope="module")
def rich_city():
    """5 años de histórico horario completo (2015-2019) + filas FUTURAS (2021) para probar fugas."""
    from backend.app.services.analysis.normality import clear_cache

    past = make_city_rows("2015-01-01", "2020-01-01")
    # temperaturas absurdas (40 °C de media): delatarían cualquier fuga de datos futuros
    future = make_city_rows("2021-06-01", "2021-07-01", base=40.0, seed=99)
    with _city(pd.concat([past, future], ignore_index=True)) as (slug, loc_id):
        clear_cache()
        yield slug, loc_id, past
    clear_cache()


@pytest.fixture
def city_factory() -> Callable:
    """Fábrica de ciudades vacías o con datos; se limpian al terminar la prueba."""
    stack: list = []

    def make(df: pd.DataFrame | None = None, tz: str = "UTC"):
        cm = _city(df, tz)
        slug, loc_id = cm.__enter__()
        stack.append(cm)
        return slug, loc_id

    yield make
    for cm in reversed(stack):
        cm.__exit__(None, None, None)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)
