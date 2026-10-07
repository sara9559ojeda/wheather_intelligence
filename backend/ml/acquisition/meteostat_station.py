"""Descarga de observaciones horarias reales de una estación (Meteostat).

Fuente de contraste frente al reanálisis ERA5 (ver fase-1 §7.2). Meteostat
agrega datos de fuentes gubernamentales (NOAA, DWD, AEMET…) a partir de
informes METAR/SYNOP. Licencia de los datos: CC BY 4.0; librería: MIT.

A diferencia de ERA5, una estación real tiene huecos, cambios de instrumento
y periodos sin datos → material útil para demostrar limpieza en la fase 3.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime

import pandas as pd
import requests

# La librería meteostat 1.6.x emite FutureWarnings de pandas al parsear su CSV
# interno; son de la librería, no de nuestro código.
warnings.filterwarnings("ignore", category=FutureWarning, module="meteostat")

from meteostat import Hourly  # noqa: E402

log = logging.getLogger(__name__)

_STATION_META_URL = (
    "https://raw.githubusercontent.com/meteostat/weather-stations/master/stations/{sid}.json"
)

# Nombres cortos de Meteostat → nombre descriptivo (para el diccionario de datos).
METEOSTAT_COLUMNS: dict[str, str] = {
    "temp": "air_temperature_c",
    "dwpt": "dew_point_c",
    "rhum": "relative_humidity_pct",
    "prcp": "precipitation_mm",
    "snow": "snow_depth_mm",
    "wdir": "wind_direction_deg",
    "wspd": "wind_speed_kmh",
    "wpgt": "wind_gust_kmh",
    "pres": "sea_level_pressure_hpa",
    "tsun": "sunshine_minutes",
    "coco": "weather_condition_code",
}


def fetch_station_metadata(station_id: str) -> dict:
    """Metadatos de la estación (nombre, país, lat/lon, elevación, identificadores)."""
    url = _STATION_META_URL.format(sid=station_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_station_hourly(station_id: str, start: date, end: date) -> pd.DataFrame:
    """Serie horaria de la estación en [start, end].

    Devuelve un DataFrame con ``timestamp`` (UTC, tz-aware) como columna y los
    nombres de Meteostat SIN renombrar (el renombrado se hace en la fase 3,
    para mantener el crudo fiel a la fuente).
    """
    start_dt = datetime(start.year, start.month, start.day)
    end_dt = datetime(end.year, end.month, end.day, 23, 59)

    log.info("Meteostat: descargando estación %s de %s a %s…", station_id, start, end)
    frame = Hourly(station_id, start_dt, end_dt).fetch()

    if frame.empty:
        raise RuntimeError(f"Meteostat no devolvió datos para la estación {station_id}")

    frame = frame.reset_index().rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame
