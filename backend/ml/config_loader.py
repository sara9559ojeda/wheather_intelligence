"""Carga de configuración y rutas del proyecto.

Un único lugar donde se resuelven:
  - la raíz del repositorio y las carpetas de datos
  - el contenido de ``backend/ml/config/location.toml``

Así ningún script hardcodea coordenadas, fechas ni rutas absolutas.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

# backend/ml/config_loader.py  ->  parents[2] == raíz del repo
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "backend" / "ml" / "artifacts"
CACHE_DIR: Path = INTERIM_DIR / "acquisition_cache"

_CONFIG_PATH: Path = PROJECT_ROOT / "backend" / "ml" / "config" / "location.toml"


@dataclass(frozen=True)
class LocationConfig:
    """Vista tipada de ``location.toml`` (solo los campos que usan los scripts)."""

    name: str
    slug: str
    latitude: float
    longitude: float
    timezone: str
    country: str

    historical_start_date: date
    era5_lag_days: int
    hourly_variables: tuple[str, ...]

    meteostat_station_id: str

    archive_base_url: str
    forecast_base_url: str
    request_timeout_seconds: int
    retry_attempts: int
    polite_delay_seconds: float

    @property
    def historical_end_date(self) -> date:
        """Último día disponible en ERA5: hoy menos el retraso de reanálisis."""
        return date.today() - timedelta(days=self.era5_lag_days)


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> LocationConfig:
    """Lee y valida ``location.toml``. Cacheado: se lee una sola vez por proceso."""
    cfg_path = path or _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {cfg_path}")

    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    loc = raw["location"]
    hist = raw["historical"]
    meteo = raw["meteostat"]
    api = raw["api"]

    return LocationConfig(
        name=loc["name"],
        slug=loc["slug"],
        latitude=float(loc["latitude"]),
        longitude=float(loc["longitude"]),
        timezone=loc["timezone"],
        country=loc["country"],
        historical_start_date=date.fromisoformat(hist["start_date"]),
        era5_lag_days=int(hist["era5_lag_days"]),
        hourly_variables=tuple(hist["hourly_variables"]),
        meteostat_station_id=str(meteo["station_id"]),
        archive_base_url=api["archive_base_url"],
        forecast_base_url=api["forecast_base_url"],
        request_timeout_seconds=int(api["request_timeout_seconds"]),
        retry_attempts=int(api["retry_attempts"]),
        polite_delay_seconds=float(api["polite_delay_seconds"]),
    )


def ensure_data_dirs() -> None:
    """Crea las carpetas de datos si no existen (idempotente)."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
