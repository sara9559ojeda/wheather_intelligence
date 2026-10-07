"""Adaptador de la Open-Meteo Forecast API (datos en tiempo real).

Sin API key. Licencia de los datos: CC BY 4.0. Las variables coinciden con las
del histórico ERA5 → las features de inferencia son las mismas que las de
entrenamiento (sin *train-serving skew*).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from backend.app.services.ingestion.weather_provider import ObservationPayload

log = logging.getLogger(__name__)

# Variables disponibles en el endpoint `current` de la Forecast API.
_CURRENT_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "surface_pressure", "pressure_msl", "precipitation", "rain", "snowfall",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "weather_code", "is_day",
]
_HOURLY_VARS = _CURRENT_VARS  # el endpoint hourly acepta el mismo conjunto

_INT_FIELDS = {"weather_code"}
_BOOL_FIELDS = {"is_day"}


def _coerce(payload: dict) -> ObservationPayload:
    out: ObservationPayload = {}
    for k, v in payload.items():
        if k == "time":
            out["observed_at"] = pd.to_datetime(v, utc=True).to_pydatetime()
        elif k == "interval":
            continue
        elif v is None:
            out[k] = None
        elif k in _INT_FIELDS:
            out[k] = int(v)
        elif k in _BOOL_FIELDS:
            out[k] = bool(v)
        else:
            out[k] = float(v)
    return out


class OpenMeteoForecastClient:
    def __init__(
        self,
        *,
        base_url: str,
        latitude: float,
        longitude: float,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url
        self.latitude = latitude
        self.longitude = longitude
        self.timeout = timeout_seconds

    def _get(self, params: dict) -> dict:
        params = {"latitude": self.latitude, "longitude": self.longitude,
                  "timezone": "UTC", **params}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(self.base_url, params=params)
            resp.raise_for_status()
            return resp.json()

    def get_current(self) -> ObservationPayload:
        data = self._get({"current": ",".join(_CURRENT_VARS)})
        return _coerce(data["current"])

    def get_recent_hourly(self, past_days: int = 3) -> list[ObservationPayload]:
        """Serie horaria continua de los últimos ``past_days`` días hasta la hora actual.

        Se pide ``forecast_days=1`` para que la serie llegue hasta "ahora" sin
        huecos (las horas aún no completadas traen el valor previsto de
        Open-Meteo); luego se recortan las horas futuras.
        """
        data = self._get({
            "hourly": ",".join(_HOURLY_VARS),
            "past_days": past_days,
            "forecast_days": 1,
        })
        h = data["hourly"]
        cutoff = datetime.now(UTC) + timedelta(minutes=90)
        rows: list[ObservationPayload] = []
        for i, t in enumerate(h["time"]):
            record = {"time": t, **{v: h[v][i] for v in _HOURLY_VARS if v in h}}
            payload = _coerce(record)
            if payload.get("observed_at") and payload["observed_at"] <= cutoff:
                rows.append(payload)
        return rows
