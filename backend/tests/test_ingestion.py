"""Pruebas de la ingesta en tiempo real (fase 6)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.app.services.ingestion.realtime import _clip, _floor_hour
from backend.app.services.ingestion.weather_provider import ObservationPayload


def test_floor_hour():
    dt = datetime(2026, 9, 9, 14, 37, 12, tzinfo=UTC)
    assert _floor_hour(dt) == datetime(2026, 9, 9, 14, 0, 0, tzinfo=UTC)


def test_clip_bounds_invalid_values():
    payload: ObservationPayload = {
        "temperature_2m": 22.0,
        "relative_humidity_2m": 130.0,   # imposible -> 100
        "wind_speed_10m": -5.0,          # imposible -> 0
        "cloud_cover": 50.0,
    }
    out = _clip(payload)
    assert out["relative_humidity_2m"] == 100.0
    assert out["wind_speed_10m"] == 0.0
    assert out["temperature_2m"] == 22.0


class _FakeProvider:
    def __init__(self, current_time: str) -> None:
        self._t = current_time

    def get_current(self) -> ObservationPayload:
        return {
            "observed_at": datetime.fromisoformat(self._t),
            "temperature_2m": 21.0, "relative_humidity_2m": 55.0,
            "surface_pressure": 945.0, "wind_speed_10m": 6.0,
            "cloud_cover": 20.0, "shortwave_radiation": 100.0, "precipitation": 0.0,
        }

    def get_recent_hourly(self, past_days: int = 3) -> list[ObservationPayload]:
        return []


@pytest.mark.db
def test_ingest_current_is_hour_aligned_and_idempotent():
    from sqlalchemy import func, select

    from backend.app.db.models import WeatherObservation
    from backend.app.db.queries import get_location
    from backend.app.db.session import session_scope
    from backend.app.services.ingestion.realtime import ingest_current

    provider = _FakeProvider("2027-01-01T09:41:00+00:00")  # fecha lejana, no colisiona
    with session_scope() as s:
        loc = get_location(s, "madrid_barajas")
        assert loc is not None

        def count() -> int:
            return s.scalar(
                select(func.count()).select_from(WeatherObservation).where(
                    WeatherObservation.observed_at
                    == datetime(2027, 1, 1, 9, 0, 0, tzinfo=UTC)
                )
            )

        ingest_current(s, provider, "madrid_barajas")
        assert count() == 1  # alineado a las 09:00

        ingest_current(s, provider, "madrid_barajas")
        assert count() == 1  # no duplica

        s.query(WeatherObservation).filter(
            WeatherObservation.observed_at
            == datetime(2027, 1, 1, 9, 0, 0, tzinfo=UTC)
        ).delete()
