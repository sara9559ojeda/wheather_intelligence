"""Modelo de datos de Weather Intelligence (ver docs/fase-1 §11).

Tablas:
  locations             catálogo de ubicaciones
  weather_observations   observaciones meteorológicas (históricas + tiempo real)
  model_runs             entrenamientos de modelos de regresión + sus métricas
  predictions            predicciones de temperatura generadas por un model_run
  cluster_models         modelos de clustering (K-Means) entrenados
  cluster_assignments    asignación de cada observación a un cluster
  anomalies              observaciones evaluadas por el detector de anomalías
  ai_analyses            interpretaciones en lenguaje natural generadas por Claude

Convenciones:
  - todos los timestamps son timezone-aware y se almacenan en UTC
  - claves primarias sintéticas (BIGINT identity en tablas de hechos, INT en catálogos)
  - medidas en NUMERIC para evitar sorpresas de coma flotante
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

# Valores permitidos para weather_observations.source
SOURCE_HISTORICAL = "historical"
SOURCE_REALTIME = "realtime"
ALLOWED_SOURCES = (SOURCE_HISTORICAL, SOURCE_REALTIME)


def _ts(**kw) -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), **kw)


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    slug: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)
    latitude: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    longitude: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    country: Mapped[str | None] = mapped_column(String(2))
    elevation_m: Mapped[float | None] = mapped_column(Numeric(6, 1))
    created_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)

    observations: Mapped[list[WeatherObservation]] = relationship(back_populates="location")


class WeatherObservation(Base):
    __tablename__ = "weather_observations"
    __table_args__ = (
        UniqueConstraint("location_id", "observed_at", "source"),
        CheckConstraint(
            f"source in {ALLOWED_SOURCES}", name="source_allowed"
        ),
        Index("ix_weather_observations_location_id_observed_at", "location_id", "observed_at"),
        Index("ix_weather_observations_source", "source"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"), nullable=False
    )
    observed_at: Mapped[datetime] = _ts(nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)

    # --- medidas (nullable: la fuente en tiempo real puede no traer todas) ---
    temperature_2m: Mapped[float | None] = mapped_column(Numeric(5, 2))
    relative_humidity_2m: Mapped[float | None] = mapped_column(Numeric(5, 2))
    dew_point_2m: Mapped[float | None] = mapped_column(Numeric(5, 2))
    apparent_temperature: Mapped[float | None] = mapped_column(Numeric(5, 2))
    surface_pressure: Mapped[float | None] = mapped_column(Numeric(6, 2))
    pressure_msl: Mapped[float | None] = mapped_column(Numeric(6, 2))
    precipitation: Mapped[float | None] = mapped_column(Numeric(6, 2))
    rain: Mapped[float | None] = mapped_column(Numeric(6, 2))
    snowfall: Mapped[float | None] = mapped_column(Numeric(6, 2))
    cloud_cover: Mapped[float | None] = mapped_column(Numeric(5, 2))
    cloud_cover_low: Mapped[float | None] = mapped_column(Numeric(5, 2))
    cloud_cover_mid: Mapped[float | None] = mapped_column(Numeric(5, 2))
    cloud_cover_high: Mapped[float | None] = mapped_column(Numeric(5, 2))
    wind_speed_10m: Mapped[float | None] = mapped_column(Numeric(6, 2))
    wind_direction_10m: Mapped[float | None] = mapped_column(Numeric(5, 1))
    wind_gusts_10m: Mapped[float | None] = mapped_column(Numeric(6, 2))
    wind_speed_100m: Mapped[float | None] = mapped_column(Numeric(6, 2))
    shortwave_radiation: Mapped[float | None] = mapped_column(Numeric(7, 2))
    direct_radiation: Mapped[float | None] = mapped_column(Numeric(7, 2))
    diffuse_radiation: Mapped[float | None] = mapped_column(Numeric(7, 2))
    et0_fao_evapotranspiration: Mapped[float | None] = mapped_column(Numeric(5, 3))
    weather_code: Mapped[int | None] = mapped_column(SmallInteger)
    is_day: Mapped[bool | None] = mapped_column(Boolean)

    ingested_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)

    location: Mapped[Location] = relationship(back_populates="observations")


class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (
        # Índice parcial: solo interesan los modelos activos ("campeón" en producción).
        Index("ix_model_runs_is_active", "is_active", postgresql_where=text("is_active")),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_type: Mapped[str] = mapped_column(String(40), nullable=False)
    horizon_hours: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    target: Mapped[str] = mapped_column(String(40), nullable=False, default="temperature_2m")

    trained_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)
    train_period: Mapped[str | None] = mapped_column(String(60))
    valid_period: Mapped[str | None] = mapped_column(String(60))
    test_period: Mapped[str | None] = mapped_column(String(60))

    # métricas de VALIDACIÓN (base de la selección del modelo)
    mae: Mapped[float | None] = mapped_column(Numeric(8, 4))
    rmse: Mapped[float | None] = mapped_column(Numeric(8, 4))
    r2: Mapped[float | None] = mapped_column(Numeric(6, 4))
    mape: Mapped[float | None] = mapped_column(Numeric(8, 4))

    hyperparams: Mapped[dict | None] = mapped_column(JSONB)
    feature_list: Mapped[list | None] = mapped_column(JSONB)
    # bloque estructurado: {"validation": {...}, "test": {...}, "cv": {...}, "skill": {...}}
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    artifact_path: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="model_run")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_run_id", "base_time"),
        Index("ix_predictions_location_id_target_time", "location_id", "target_time"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"), nullable=False
    )
    model_run_id: Mapped[int] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False
    )
    base_time: Mapped[datetime] = _ts(nullable=False)
    target_time: Mapped[datetime] = _ts(nullable=False)
    predicted_temperature: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    lower_bound: Mapped[float | None] = mapped_column(Numeric(5, 2))
    upper_bound: Mapped[float | None] = mapped_column(Numeric(5, 2))
    actual_temperature: Mapped[float | None] = mapped_column(Numeric(5, 2))
    created_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)

    model_run: Mapped[ModelRun] = relationship(back_populates="predictions")


class ClusterModel(Base):
    __tablename__ = "cluster_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    k: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    silhouette: Mapped[float | None] = mapped_column(Numeric(6, 4))
    trained_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)
    feature_list: Mapped[list | None] = mapped_column(JSONB)
    centroids: Mapped[dict | None] = mapped_column(JSONB)
    cluster_labels: Mapped[dict | None] = mapped_column(JSONB)
    artifact_path: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    assignments: Mapped[list[ClusterAssignment]] = relationship(back_populates="cluster_model")


class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"
    __table_args__ = (
        UniqueConstraint("cluster_model_id", "observation_id"),
        Index(
            "ix_cluster_assignments_cluster_model_id_cluster_id",
            "cluster_model_id",
            "cluster_id",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    cluster_model_id: Mapped[int] = mapped_column(
        ForeignKey("cluster_models.id", ondelete="CASCADE"), nullable=False
    )
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("weather_observations.id", ondelete="CASCADE"), nullable=False
    )
    cluster_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    distance_to_centroid: Mapped[float | None] = mapped_column(Numeric(10, 4))

    cluster_model: Mapped[ClusterModel] = relationship(back_populates="assignments")


class Anomaly(Base):
    __tablename__ = "anomalies"
    __table_args__ = (
        UniqueConstraint("observation_id", "detector"),
        Index("ix_anomalies_location_id_observed_at", "location_id", "observed_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"), nullable=False
    )
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("weather_observations.id", ondelete="CASCADE"), nullable=False
    )
    observed_at: Mapped[datetime] = _ts(nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    detector: Mapped[str] = mapped_column(String(40), nullable=False, default="isolation_forest")
    features_used: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)


class RetrainingRun(Base):
    """Un intento de reentrenamiento (fase 9) — promovido o no, para trazabilidad."""

    __tablename__ = "retraining_runs"
    __table_args__ = (
        Index("ix_retraining_runs_location_id_triggered_at", "location_id", "triggered_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"), nullable=False
    )
    triggered_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)
    # volume | calendar | manual
    trigger_reason: Mapped[str] = mapped_column(String(40), nullable=False)
    n_new_observations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data_cutoff: Mapped[datetime | None] = _ts(nullable=True)
    # completed | not_promoted | aborted_validation | aborted_error
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    validation_report: Mapped[dict | None] = mapped_column(JSONB)
    # {horizon: {challenger_type, challenger_rmse, champion_rmse_on_new_test,
    #            champion_original_rmse, promoted, reason}}
    results: Mapped[dict | None] = mapped_column(JSONB)
    duration_seconds: Mapped[float | None] = mapped_column(Numeric(8, 2))


class AiAnalysis(Base):
    __tablename__ = "ai_analyses"
    __table_args__ = (
        Index("ix_ai_analyses_location_id_generated_at", "location_id", "generated_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id", ondelete="RESTRICT"), nullable=False
    )
    generated_at: Mapped[datetime] = _ts(server_default=func.now(), nullable=False)
    period: Mapped[str | None] = mapped_column(String(40))
    input_summary: Mapped[dict | None] = mapped_column(JSONB)
    model: Mapped[str | None] = mapped_column(String(60))
    output_text: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
