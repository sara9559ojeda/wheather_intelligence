"""Esquemas Pydantic de las funcionalidades de análisis contextual.

Cubren las respuestas de ``/api/analysis/*``, ``/api/evaluation/*`` y
``/api/forecast/*``. Todos los campos que pueden faltar son opcionales: la API
devuelve un estado explícito (``insufficient``/``unavailable``) en vez de inventar.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

# ------------------------------------------------------------------ normalidad
AnalysisStatus = Literal["ok", "partial", "insufficient", "unavailable"]


class ReferenceInfo(BaseModel):
    """Con qué observaciones históricas se comparó y qué tan representativas son."""

    description: str
    n_observations: int
    period_start: datetime | None = None
    period_end: datetime | None = None
    years: int | None = None
    window_step: int                    # 0 = ventana más estricta; >0 = se amplió
    representativeness: Literal["alta", "media", "baja"]
    widened: bool                       # True si hubo que ampliar la ventana
    min_sample: int
    local_month: int
    local_hour: int
    timezone: str


class VariableNormality(BaseModel):
    variable: str
    label: str
    unit: str
    status: Literal["ok", "insufficient", "not_applicable"]
    value: float | None = None
    percentile: float | None = None     # % de la referencia con valor <= actual
    classification: str | None = None   # clave estable (typical, high, exceptional_high…)
    classification_label: str | None = None
    is_extreme: bool = False            # extremo UNIVARIABLE (fuera del 2,5–97,5 %)
    median: float | None = None
    usual_range: tuple[float, float] | None = None     # p10–p90
    typical_range: tuple[float, float] | None = None   # p25–p75
    n_sample: int = 0
    wet_fraction: float | None = None   # solo precipitación
    explanation: str


class ContributorOut(BaseModel):
    variable: str
    label: str
    percentile: float
    direction: Literal["above", "below"]


class AnomalyInfo(BaseModel):
    available: bool
    is_multivariate_anomaly: bool | None = None
    score: float | None = None
    kind: Literal["none", "multivariate", "univariate", "both"] | None = None
    contributors: list[ContributorOut] = []
    explanation: str


class SimilarInfo(BaseModel):
    available: bool
    n_similar: int = 0
    share_pct: float | None = None
    years_with_similar: int | None = None
    years_total: int | None = None
    tolerance: str | None = None
    explanation: str


class NormalityOut(BaseModel):
    status: AnalysisStatus
    location: str
    slug: str
    observed_at: datetime | None = None
    reference: ReferenceInfo | None = None
    variables: list[VariableNormality] = []
    anomaly: AnomalyInfo | None = None
    similar: SimilarInfo | None = None
    summary: str
    notes: list[str] = []


# ------------------------------------------------------------------ regímenes
class DominantVariable(BaseModel):
    variable: str
    label: str
    unit: str
    direction: Literal["above", "below"]
    z: float
    median: float


class SeasonShare(BaseModel):
    month: int
    share_pct: float            # % de las horas de este régimen que caen en ese mes
    lift: float                 # cuánto más/menos frecuente que el reparto uniforme del mes


class PersistenceInfo(BaseModel):
    n_episodes: int
    median_hours: float
    mean_hours: float
    p90_hours: float
    stay_probability_1h: float | None = None


class RegimeProfile(BaseModel):
    regime_id: int
    label: str
    description: str
    n_observations: int
    share_pct: float
    medians: dict[str, float]
    iqr: dict[str, tuple[float, float]]
    z_vs_city: dict[str, float]         # desviación del régimen respecto al clima de la ciudad
    dominant: list[DominantVariable]
    wet_share_pct: float
    seasonality: list[SeasonShare]
    peak_months: list[int]
    persistence: PersistenceInfo | None = None
    distance_p50: float | None = None
    distance_p90: float | None = None
    distance_p99: float | None = None


class CurrentRegime(BaseModel):
    regime_id: int
    label: str
    distance_to_centroid: float
    distance_band: Literal["típica", "algo alejada", "atípica"]
    second_regime_id: int | None = None
    second_label: str | None = None
    distance_ratio_to_second: float | None = None   # d1/d2 (cerca de 1 = frontera difusa)
    observed_at: datetime
    explanation: str


class TimelinePoint(BaseModel):
    t: datetime
    regime_id: int


class Episode(BaseModel):
    regime_id: int
    label: str
    start: datetime
    end: datetime
    hours: int


class TransitionOut(BaseModel):
    to_regime_id: int
    label: str
    count: int
    probability: float          # frecuencia histórica entre cambios de régimen


class RegimesOut(BaseModel):
    status: AnalysisStatus
    location: str
    slug: str
    model: dict | None = None
    current: CurrentRegime | None = None
    regimes: list[RegimeProfile] = []
    timeline: list[TimelinePoint] = []
    episodes: list[Episode] = []
    recent_share_pct: dict[int, float] = {}
    transitions_from_current: list[TransitionOut] = []
    transitions_supported: bool = False
    transitions_total: int = 0
    notes: list[str] = []


# ------------------------------------------------------------------ evaluación
class BaselineMetrics(BaseModel):
    n: int
    mae: float
    rmse: float


class HorizonEvaluation(BaseModel):
    horizon_hours: int
    n_verified: int
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None                       # error medio firmado (predicho − observado)
    persistence: BaselineMetrics | None = None
    skill_vs_persistence: float | None = None       # 1 − RMSE_modelo / RMSE_persistencia
    reference: BaselineMetrics | None = None        # pronóstico archivado de Open-Meteo
    model_on_reference_rows: BaselineMetrics | None = None   # modelo en las MISMAS filas
    coverage_pct: float | None = None
    mean_interval_width: float | None = None
    last_verified_at: datetime | None = None
    model_versions: list[str] = []
    model_types: list[str] = []
    explanation: str


class SeriesPoint(BaseModel):
    day: str
    n: int
    mae: float
    rmse: float


class HorizonSeries(BaseModel):
    horizon_hours: int
    points: list[SeriesPoint]


class OfflineModel(BaseModel):
    horizon_hours: int
    model_type: str
    is_champion: bool
    test_rmse: float | None = None
    test_mae: float | None = None
    valid_rmse: float | None = None
    trained_at: datetime | None = None


class PairPoint(BaseModel):
    target_time: datetime
    predicted: float
    observed: float
    lower: float | None = None
    upper: float | None = None


class EvaluationOut(BaseModel):
    status: Literal["ok", "collecting", "unavailable"]
    location: str
    slug: str
    window_days: int
    generated_at: datetime
    n_logged_total: int
    n_verified_total: int
    n_pending: int
    first_logged_at: datetime | None = None
    last_verified_at: datetime | None = None
    horizons: list[HorizonEvaluation] = []
    series: list[HorizonSeries] = []
    offline_models: list[OfflineModel] = []
    notes: list[str] = []


class PairsOut(BaseModel):
    horizon_hours: int
    window_days: int
    n_total: int
    points: list[PairPoint]


# ------------------------------------------------------------------ incertidumbre
class ReliabilityOut(BaseModel):
    horizon_hours: int
    predicted_temperature: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_width: float | None = None
    nominal_level: float | None = None
    interval_method: str | None = None
    base_time: datetime | None = None
    target_time: datetime | None = None
    live_coverage_pct: float | None = None
    live_n: int = 0
    live_mean_width: float | None = None
    offline_coverage_pct: float | None = None
    offline_n: int | None = None
    offline_mean_width: float | None = None
    level: Literal["alta", "media", "baja", "preliminar", "no_evaluable"]
    level_label: str
    degraded: bool = False
    reasons: list[str] = []
    statement: str                      # frase para el usuario
    coverage_statement: str


class ForecastReliabilityOut(BaseModel):
    status: Literal["ok", "unavailable"]
    location: str
    slug: str
    generated_at: datetime
    horizons: list[ReliabilityOut] = []
    criteria: list[str] = []
    notes: list[str] = []


# ------------------------------------------------------------------ analista
class AnalystQuestion(BaseModel):
    id: str
    text: str


class AnalystAsk(BaseModel):
    question: str | None = None
    intent: str | None = None
    detail: bool = False


class AnalystSection(BaseModel):
    kind: Literal["datos", "asociacion", "hipotesis"]
    text: str


class EvidenceItem(BaseModel):
    label: str
    value: str
    source: str


class AnalystAnswer(BaseModel):
    mode: Literal["ai", "deterministic"]
    intent: str
    question: str
    answer: str
    sections: list[AnalystSection] = []
    deterministic_answer: str
    evidence: list[EvidenceItem] = []
    limitations: list[str] = []
    availability: dict[str, bool] = {}
    fallback_reason: str | None = None
    model: str | None = None
    generated_at: datetime
    context_as_of: datetime | None = None
