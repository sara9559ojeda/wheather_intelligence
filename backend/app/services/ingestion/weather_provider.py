"""Interfaz de proveedor de datos meteorológicos en tiempo real (patrón adapter).

Cambiar de Open-Meteo a otro proveedor = una implementación nueva de este
Protocol, no reescribir el resto del sistema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, TypedDict


class ObservationPayload(TypedDict, total=False):
    """Observación normalizada (claves = columnas de weather_observations)."""

    observed_at: datetime
    temperature_2m: float | None
    relative_humidity_2m: float | None
    dew_point_2m: float | None
    apparent_temperature: float | None
    surface_pressure: float | None
    pressure_msl: float | None
    precipitation: float | None
    rain: float | None
    snowfall: float | None
    cloud_cover: float | None
    wind_speed_10m: float | None
    wind_direction_10m: float | None
    wind_gusts_10m: float | None
    shortwave_radiation: float | None
    weather_code: int | None
    is_day: bool | None


class WeatherProvider(Protocol):
    def get_current(self) -> ObservationPayload:
        """Condiciones actuales."""
        ...

    def get_recent_hourly(self, past_days: int = 3) -> list[ObservationPayload]:
        """Serie horaria de los últimos ``past_days`` días (para el backfill inicial)."""
        ...
