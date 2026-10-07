"""Esquemas de respuesta de la API (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ObservationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    observed_at: datetime
    source: str
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    dew_point_2m: float | None = None
    apparent_temperature: float | None = None
    surface_pressure: float | None = None
    pressure_msl: float | None = None
    precipitation: float | None = None
    cloud_cover: float | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    wind_gusts_10m: float | None = None
    shortwave_radiation: float | None = None
    weather_code: int | None = None
    is_day: bool | None = None


class CurrentWeatherOut(BaseModel):
    location: str
    latitude: float
    longitude: float
    observation: ObservationOut
    age_minutes: float
    delayed: bool


class StatVariable(BaseModel):
    variable: str
    mean: float
    median: float
    std: float
    min: float
    p05: float
    p25: float
    p75: float
    p95: float
    max: float


class StatsOut(BaseModel):
    location: str
    period: str
    n_rows: int
    variables: list[StatVariable]


class AnomalyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    observed_at: datetime
    anomaly_score: float
    is_anomaly: bool
    detector: str
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    surface_pressure: float | None = None
    wind_speed_10m: float | None = None


class ClusterOut(BaseModel):
    cluster_id: int
    label: str
    share_pct: float
    centroid: dict[str, float]


class ClusterModelOut(BaseModel):
    k: int
    silhouette: float | None
    trained_at: datetime
    clusters: list[ClusterOut]


class HorizonPredictionOut(BaseModel):
    horizon_hours: int
    base_time: datetime
    target_time: datetime
    predicted_temperature: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    actual_temperature: float | None = None
    model_type: str


class ModelRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model_type: str
    horizon_hours: int
    target: str
    is_active: bool
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    trained_at: datetime
    metrics: dict | None = None


class PercentileInfo(BaseModel):
    variable: str
    value: float
    percentile: float
    normal_range: tuple[float, float]


class HistoricalComparisonOut(BaseModel):
    location: str
    reference: str
    observed_at: datetime
    percentiles: list[PercentileInfo]
    cluster_id: int | None = None
    cluster_label: str | None = None
    distance_to_centroid: float | None = None
    is_anomaly: bool | None = None
    anomaly_score: float | None = None


class RetrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    triggered_at: datetime
    trigger_reason: str
    n_new_observations: int
    data_cutoff: datetime | None = None
    status: str
    duration_seconds: float | None = None
    validation_report: dict | None = None
    results: dict | None = None


class DashboardOut(BaseModel):
    location: str
    latitude: float
    longitude: float
    generated_at: datetime
    current: CurrentWeatherOut
    predictions: list[HorizonPredictionOut]
    historical_comparison: HistoricalComparisonOut
    recent_anomalies: list[AnomalyOut]
    clusters: ClusterModelOut
    model_performance: list[ModelRunOut]
    ai_insight: str | None = None
