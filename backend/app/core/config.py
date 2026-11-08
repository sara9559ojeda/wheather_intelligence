"""Configuración tipada del backend, leída de ``.env`` (nunca hardcodeada).

Uso:
    from backend.app.core.config import settings
    settings.database_url
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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
    # "queue" (por defecto): pool de conexiones clásico, para un proceso de larga vida.
    # "null": sin pool (NullPool), una conexión por sesión — para funciones serverless
    #         (Vercel) detrás del pooler de Neon/PgBouncer.
    db_pool_mode: Literal["queue", "null"] = "queue"

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
    # Analista (preguntas contextuales)
    analyst_timeout_seconds: float = 25.0    # tiempo máximo de la llamada a Claude
    analyst_ai_per_hour: int = 20            # llamadas a Claude por usuario y hora
    analyst_asks_per_hour: int = 120         # preguntas totales (con o sin IA) por usuario y hora
    analyst_cache_minutes: int = 10          # respuestas de IA a preguntas sugeridas

    # --- Open-Meteo ---
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"

    # --- Ingesta / API ---
    ingest_interval_minutes: int = 30
    backend_cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    # --- Modo de ejecución (proceso de larga vida vs. serverless) ---
    # En Docker/local ambos a true (comportamiento de siempre). En Vercel a false:
    # la ingesta la dispara GitHub Actions contra POST /api/tasks/ingest.
    enable_scheduler: bool = True       # APScheduler (ingesta + comprobación de reentreno)
    run_startup_backfill: bool = True   # backfill de 72 h + ingesta al arrancar

    # --- Tiempo real ---
    # Vacío: bus de eventos en memoria (una instancia). Con valor (redis:// o rediss://):
    # pub/sub de Redis, necesario con varias instancias (Vercel + Upstash).
    redis_url: str = ""
    # Cierre ordenado de cada WebSocket tras N segundos (0 = sin límite). En Vercel,
    # algo menos que la duración máxima de la función (p. ej. 280 con maxDuration=300).
    ws_max_connection_seconds: int = 0

    # --- Autenticación de usuarios (login del dashboard) ---
    # Secreto con el que se firman los JWT. Vacío = login deshabilitado (503) y la API
    # de datos queda cerrada: nunca abierta. Genera uno largo, p. ej.:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    auth_secret_key: str = ""
    auth_token_minutes: int = 720        # duración de la sesión (12 h)
    # Registro público de cuentas. En producción, ponlo a false y crea los usuarios
    # con `python -m backend.app.create_user`.
    allow_registration: bool = True

    # --- Endpoint protegido de tareas (POST /api/tasks/ingest) ---
    # Vacío = endpoint deshabilitado (503). Se envía como "Authorization: Bearer <token>".
    ingest_token: str = ""

    # --- Reentrenamiento (fase 9) ---
    retrain_check_interval_hours: int = 24     # cada cuánto se COMPRUEBA si toca reentrenar
    retrain_volume_trigger: int = 1000         # nº de observaciones realtime nuevas que disparan
    retrain_calendar_days: int = 30            # o llevar este nº de días sin comprobar
    retrain_epsilon: float = 0.02              # el retador debe mejorar el RMSE al menos un 2 %
    retrain_degradation_threshold: float = 1.20  # o el campeón haber empeorado un 20 %

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, v: str) -> str:
        """Acepta URLs ``postgres://`` / ``postgresql://`` (las que da Neon/Vercel) y
        fuerza el driver psycopg 3, que es el instalado."""
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
