"""Pruebas del ETL (fase 2).

- ``validate`` y ``_to_records`` son puras → se prueban siempre (offline).
- El round-trip contra PostgreSQL va marcado con ``@pytest.mark.db``:
    pytest -m "not db"     # omite
    pytest -m db           # solo estas (requiere docker compose up -d db + alembic upgrade head)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from backend.app.services.etl.historical_loader import _to_records, validate


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T02:00"], utc=True
            ),
            "temperature_2m": [3.1, 2.6, float("nan")],
            "relative_humidity_2m": [71, 74, 80],
            "surface_pressure": [945.0, 946.1, 999.9],   # 999.9 fuera de rango
            "wind_direction_10m": [270, 268, 265],
            "weather_code": [0, 1, 61],
            "is_day": [0, 0, 1],
        }
    )


def test_validate_counts_issues():
    rep = validate(_sample_df())
    assert rep.n_rows == 3
    assert rep.duplicate_timestamps == 0
    assert rep.unsorted is False
    assert rep.nulls["temperature_2m"] == 1
    assert rep.out_of_range["surface_pressure"] == 1
    assert rep.out_of_domain["weather_code"] == 0
    assert rep.out_of_domain["is_day"] == 0


def test_to_records_types_and_nan_handling():
    records = _to_records(_sample_df(), location_id=7, source="historical")
    assert len(records) == 3
    r0, _, r2 = records
    assert r0["location_id"] == 7
    assert r0["source"] == "historical"
    assert isinstance(r0["observed_at"], datetime)
    assert r0["observed_at"].tzinfo is not None
    assert r0["is_day"] is False
    assert r0["weather_code"] == 0
    assert r2["temperature_2m"] is None               # NaN -> None


@pytest.mark.db
def test_round_trip_is_idempotent():
    from sqlalchemy import delete, func, select

    from backend.app.db.models import Location, WeatherObservation
    from backend.app.db.session import session_scope
    from backend.app.services.etl.historical_loader import load_dataframe

    df = _sample_df()
    df["timestamp"] = pd.date_range("1990-01-01", periods=3, freq="h", tz=UTC)

    with session_scope() as s:
        loc = s.scalar(select(Location).where(Location.slug == "madrid_barajas"))
        assert loc is not None, "Ejecuta antes el ETL / seed de la ubicación"
        s.execute(
            delete(WeatherObservation).where(
                WeatherObservation.location_id == loc.id,
                WeatherObservation.observed_at < datetime(2000, 1, 1, tzinfo=UTC),
            )
        )

    def count() -> int:
        with session_scope() as s:
            return s.scalar(
                select(func.count()).select_from(WeatherObservation).where(
                    WeatherObservation.observed_at
                    < datetime(2000, 1, 1, tzinfo=UTC)
                )
            )

    with session_scope() as s:
        load_dataframe(s, df, location_id=loc.id)
    assert count() == 3

    with session_scope() as s:  # segunda carga: no duplica
        load_dataframe(s, df, location_id=loc.id)
    assert count() == 3

    with session_scope() as s:  # limpieza
        s.execute(
            delete(WeatherObservation).where(
                WeatherObservation.observed_at < datetime(2000, 1, 1, tzinfo=UTC)
            )
        )
