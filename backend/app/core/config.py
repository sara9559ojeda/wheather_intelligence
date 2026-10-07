"""Configuración tipada del backend, leída de ``.env`` (nunca hardcodeada).

Uso:
    from backend.app.core.config import settings
    settings.database_url
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[3] == raíz del repo
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Base de datos ---
    database_url: str = Field(
        default="postgresql+psycopg://weather:weather_dev_local@localhost:5432/weather_intelligence"
    )

    # --- Ubicación del análisis (coincide con backend/ml/config/location.toml) ---
    wi_location_name: str = "Madrid-Barajas"
    wi_location_slug: str = "madrid_barajas"
    wi_latitude: float = 40.4936
    wi_longitude: float = -3.5668
    wi_timezone: str = "Europe/Madrid"

    # --- Anthropic (solo backend) ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    ai_cache_minutes: int = 60       # no re-llamar a Claude si hay análisis más nuevo
    ai_min_gap_minutes: int = 30     # separación mínima entre generaciones automáticas

    # --- Open-Meteo ---
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"

    # --- Ingesta / API ---
    ingest_interval_minutes: int = 30
    backend_cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    # --- Reentrenamiento (fase 9) ---
    retrain_check_interval_hours: int = 24     # cada cuánto se COMPRUEBA si toca reentrenar
    retrain_volume_trigger: int = 1000         # nº de observaciones realtime nuevas que disparan
    retrain_calendar_days: int = 30            # o llevar este nº de días sin comprobar
    retrain_epsilon: float = 0.02              # el retador debe mejorar el RMSE al menos un 2 %
    retrain_degradation_threshold: float = 1.20  # o el campeón haber empeorado un 20 %

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
