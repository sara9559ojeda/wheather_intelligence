"""Ensamblado del resumen del dashboard y del resumen estructurado para Claude."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    AiAnalysis,
    Anomaly,
    ClusterModel,
    ModelRun,
    WeatherObservation,
)
from backend.app.db.queries import get_location, load_observations_df
from backend.app.schemas import (
    AnomalyOut,
    CurrentWeatherOut,
    DashboardOut,
    HistoricalComparisonOut,
    HorizonPredictionOut,
    ModelRunOut,
    ObservationOut,
)
from backend.app.schemas.models import ClusterModelOut, ClusterOut
from backend.app.services.artifacts import get_predictor
from backend.app.services.historical.comparison import compare

log = logging.getLogger(__name__)
_COMPARE_KEYS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
    "cloud_cover", "shortwave_radiation", "precipitation",
]


def _latest(db: Session, location_id: int) -> WeatherObservation | None:
    return db.scalar(
        select(WeatherObservation)
        .where(WeatherObservation.location_id == location_id)
        .order_by(WeatherObservation.observed_at.desc())
        .limit(1)
    )


def _predictions(db: Session, slug: str) -> list[HorizonPredictionOut]:
    predictor = get_predictor()
    if predictor is None:
        return []
    recent = load_observations_df(db, slug).sort_values("timestamp").tail(72)
    try:
        return [HorizonPredictionOut(**p.as_dict())
                for p in predictor.predict_from_observations(recent)]
    except ValueError as exc:
        log.warning("Sin predicciones para el dashboard: %s", exc)
        return []


def _clusters(db: Session) -> ClusterModelOut | None:
    cm = db.scalar(
        select(ClusterModel).where(ClusterModel.is_active).order_by(ClusterModel.id.desc())
    )
    if cm is None:
        return None
    centroids = {int(r["cluster"]): r for r in (cm.centroids or [])}
    labels = {int(k): v for k, v in (cm.cluster_labels or {}).items()}
    out = [
        ClusterOut(
            cluster_id=cid, label=labels[cid],
            share_pct=float(centroids.get(cid, {}).get("pct", 0.0)),
            centroid={k: float(v) for k, v in centroids.get(cid, {}).items()
                      if isinstance(v, int | float) and k not in ("cluster", "n", "pct")},
        )
        for cid in sorted(labels)
    ]
    return ClusterModelOut(
        k=cm.k, silhouette=float(cm.silhouette) if cm.silhouette is not None else None,
        trained_at=cm.trained_at, clusters=out,
    )


def _model_performance(db: Session) -> list[ModelRunOut]:
    rows = db.scalars(
        select(ModelRun)
        .where(ModelRun.is_active.is_(True), ModelRun.model_type.notlike("baseline%"))
        .order_by(ModelRun.horizon_hours)
    ).all()
    return [ModelRunOut.model_validate(r) for r in rows]


def _recent_anomalies(db: Session, location_id: int, limit: int = 10) -> list[AnomalyOut]:
    rows = db.execute(
        select(Anomaly, WeatherObservation)
        .join(WeatherObservation, WeatherObservation.id == Anomaly.observation_id)
        .where(Anomaly.location_id == location_id, Anomaly.is_anomaly.is_(True))
        .order_by(Anomaly.observed_at.desc())
        .limit(limit)
    ).all()
    return [
        AnomalyOut(
            observed_at=a.observed_at, anomaly_score=float(a.anomaly_score),
            is_anomaly=a.is_anomaly, detector=a.detector,
            temperature_2m=_f(o.temperature_2m),
            relative_humidity_2m=_f(o.relative_humidity_2m),
            surface_pressure=_f(o.surface_pressure),
            wind_speed_10m=_f(o.wind_speed_10m),
        )
        for a, o in rows
    ]


def _current_out(db: Session, loc, obs: WeatherObservation) -> CurrentWeatherOut:
    age = max(0.0, (datetime.now(UTC) - obs.observed_at).total_seconds() / 60)
    return CurrentWeatherOut(
        location=loc.name, latitude=float(loc.latitude), longitude=float(loc.longitude),
        observation=ObservationOut.model_validate(obs),
        age_minutes=round(age, 1), delayed=age > 120,
    )


def _latest_ai_insight(db: Session, location_id: int) -> str | None:
    """Última interpretación de Claude ya persistida (el dashboard no llama a la API)."""
    row = db.scalar(
        select(AiAnalysis)
        .where(AiAnalysis.location_id == location_id)
        .order_by(AiAnalysis.generated_at.desc())
        .limit(1)
    )
    return row.output_text if row else None


def build_dashboard(db: Session, slug: str) -> DashboardOut:
    loc = get_location(db, slug)
    if loc is None:
        raise ValueError("Ubicación no encontrada")
    obs = _latest(db, loc.id)
    if obs is None:
        raise ValueError("Sin observaciones")

    values = {k: _f(getattr(obs, k)) for k in _COMPARE_KEYS}
    comparison = compare(db, slug, obs.observed_at, values)

    return DashboardOut(
        location=loc.name, latitude=float(loc.latitude), longitude=float(loc.longitude),
        generated_at=datetime.now(UTC),
        current=_current_out(db, loc, obs),
        predictions=_predictions(db, slug),
        historical_comparison=HistoricalComparisonOut(**comparison),
        recent_anomalies=_recent_anomalies(db, loc.id),
        clusters=_clusters(db) or ClusterModelOut(k=0, silhouette=None,
                                                  trained_at=datetime.now(UTC),
                                                  clusters=[]),
        model_performance=_model_performance(db),
        ai_insight=_latest_ai_insight(db, loc.id),
    )


def build_ai_summary(db: Session, slug: str) -> dict:
    """Resumen estructurado y acotado que se enviará a Claude (fase 7)."""
    dash = build_dashboard(db, slug)
    o = dash.current.observation
    comp = dash.historical_comparison
    pct = {p.variable: p.percentile for p in comp.percentiles}
    normal = {p.variable: p.normal_range for p in comp.percentiles}

    pred = next((p for p in dash.predictions if p.horizon_hours == 6), None)
    champ6 = next((m for m in dash.model_performance if m.horizon_hours == 6), None)

    corr = _correlation_last_days(db, slug, days=30)

    return {
        "location": dash.location,
        "period": "current",
        "generated_at_utc": dash.generated_at.isoformat(),
        "current": {
            "temperature_c": o.temperature_2m,
            "humidity_pct": o.relative_humidity_2m,
            "pressure_hpa": o.surface_pressure,
            "wind_kmh": o.wind_speed_10m,
            "precipitation_mm": o.precipitation,
            "cloud_cover_pct": o.cloud_cover,
        },
        "historical_context": {
            "reference": comp.reference,
            "temperature_percentile": pct.get("temperature_2m"),
            "humidity_percentile": pct.get("relative_humidity_2m"),
            "pressure_percentile": pct.get("surface_pressure"),
            "normal_temp_range_c": normal.get("temperature_2m"),
        },
        "regime": {
            "cluster_id": comp.cluster_id,
            "label": comp.cluster_label,
            "distance_to_centroid": comp.distance_to_centroid,
        },
        "anomaly": {
            "is_anomaly": comp.is_anomaly,
            "score": comp.anomaly_score,
            "detector": "isolation_forest_on_residuals",
            "note": "valor inusual respecto al histórico; no implica fenómeno extremo",
        },
        "prediction": None if pred is None else {
            "horizon_hours": pred.horizon_hours,
            "predicted_temperature_c": pred.predicted_temperature,
            "interval_c": [pred.lower_bound, pred.upper_bound],
            "model_type": pred.model_type,
            "model_rmse_c": float(champ6.rmse) if champ6 and champ6.rmse else None,
            "beats_persistence_baseline": (
                bool(champ6.metrics.get("beats_persistence"))
                if champ6 and champ6.metrics else None
            ),
        },
        "correlations_last_30d": corr,
        "data_quality": {
            "observation_age_minutes": dash.current.age_minutes,
            "delayed": dash.current.delayed,
        },
    }


def _correlation_last_days(db: Session, slug: str, days: int = 30) -> dict:
    df = load_observations_df(db, slug)
    cutoff = datetime.now(UTC) - timedelta(days=days)
    df = df[df["timestamp"] >= cutoff.replace(tzinfo=UTC)]
    if len(df) < 50:
        return {}
    out = {}
    for a, b in (("temperature_2m", "relative_humidity_2m"),
                 ("temperature_2m", "shortwave_radiation")):
        pair = df[[a, b]].dropna()
        if len(pair) > 30:
            out[f"{a.split('_')[0]}_{b.split('_')[0]}"] = round(
                float(pair[a].corr(pair[b])), 3
            )
    return out


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 2)
