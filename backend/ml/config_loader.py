"""Carga de configuración y rutas del proyecto.

Un único lugar donde se resuelven:
  - la raíz del repositorio y las carpetas de datos
  - el contenido de ``backend/ml/config/location.toml``

Así ningún script hardcodea coordenadas, fechas ni rutas absolutas.
"""

from __future__ import annotations

import os
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
_CATALOG_PATH: Path = PROJECT_ROOT / "backend" / "ml" / "config" / "locations.toml"

# Ubicación por defecto: la de location.toml (Madrid-Barajas, la del proyecto original).
DEFAULT_SLUG: str = "madrid_barajas"


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


def active_slug() -> str:
    """Ubicación con la que trabajan los scripts del pipeline.

    Se elige con la variable de entorno ``WI_PIPELINE_LOCATION`` (la fija
    ``add_location`` al lanzar cada paso); sin ella, la ubicación por defecto.
    """
    return os.environ.get("WI_PIPELINE_LOCATION") or DEFAULT_SLUG


def load_catalog() -> dict[str, dict]:
    """Ciudades extra de ``locations.toml``, indexadas por slug (vacío si no existe)."""
    if not _CATALOG_PATH.exists():
        return {}
    with _CATALOG_PATH.open("rb") as fh:
        entries = tomllib.load(fh).get("locations", [])
    return {e["slug"]: e for e in entries}


def load_config(path: Path | None = None, slug: str | None = None) -> LocationConfig:
    """Configuración de una ubicación (por defecto, la activa del pipeline)."""
    return _load_config(path, slug or active_slug())


@lru_cache(maxsize=16)
def _load_config(path: Path | None, slug: str) -> LocationConfig:
    """Lee ``location.toml`` y, si ``slug`` es otra ciudad, la superpone desde el catálogo."""
    cfg_path = path or _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {cfg_path}")

    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    loc = dict(raw["location"])
    hist = raw["historical"]
    meteo = raw["meteostat"]
    api = raw["api"]
    start_date = hist["start_date"]
    station_id = str(meteo["station_id"])

    if slug != loc["slug"]:
        entry = load_catalog().get(slug)
        if entry is None:
            raise KeyError(f"Ubicación desconocida: {slug!r}. Añádela a {_CATALOG_PATH.name}.")
        keys = ("name", "slug", "latitude", "longitude", "timezone", "country")
        loc.update({k: entry[k] for k in keys})
        start_date = entry.get("start_date", start_date)
        station_id = ""  # el contraste con una estación Meteostat solo existe para Madrid

    return LocationConfig(
        name=loc["name"],
        slug=loc["slug"],
        latitude=float(loc["latitude"]),
        longitude=float(loc["longitude"]),
        timezone=loc["timezone"],
        country=loc["country"],
        historical_start_date=date.fromisoformat(start_date),
        era5_lag_days=int(hist["era5_lag_days"]),
        hourly_variables=tuple(hist["hourly_variables"]),
        meteostat_station_id=station_id,
        archive_base_url=api["archive_base_url"],
        forecast_base_url=api["forecast_base_url"],
        request_timeout_seconds=int(api["request_timeout_seconds"]),
        retry_attempts=int(api["retry_attempts"]),
        polite_delay_seconds=float(api["polite_delay_seconds"]),
    )


def processed_dir(slug: str | None = None) -> Path:
    """Carpeta del dataset procesado. La ubicación por defecto conserva la ruta histórica."""
    slug = slug or active_slug()
    return PROCESSED_DIR if slug == DEFAULT_SLUG else PROCESSED_DIR / slug


def model_artifact_name(horizon: int, slug: str | None = None) -> str:
    """Nombre del artefacto del modelo de temperatura a ``horizon`` horas.

    Madrid (la ubicación por defecto) mantiene ``model_temp_h{H}.joblib``;
    el resto lleva el slug: ``model_<slug>_temp_h{H}.joblib``.
    """
    slug = slug or active_slug()
    if slug == DEFAULT_SLUG:
        return f"model_temp_h{horizon}.joblib"
    return f"model_{slug}_temp_h{horizon}.joblib"


def ensure_data_dirs() -> None:
    """Crea las carpetas de datos si no existen (idempotente)."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
