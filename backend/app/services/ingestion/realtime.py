"""Ingesta en tiempo real: Open-Meteo Forecast -> weather_observations -> procesamiento.

Flujo (fase 1, §16):
    API -> validación -> upsert (source='realtime') -> features -> predicción
        -> anomalía -> comparación histórica -> evento SSE
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import Anomaly, ModelRun, Prediction, WeatherObservation
from backend.app.db.queries import get_location, load_observations_df
from backend.app.services.artifacts import get_predictor
from backend.app.services.events import bus
from backend.app.services.historical.comparison import compare
from backend.app.services.ingestion.weather_provider import ObservationPayload, WeatherProvider
from backend.ml.quality_rules import PLAUSIBLE_RANGES

log = logging.getLogger(__name__)

_MEASURE_KEYS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "surface_pressure", "pressure_msl", "precipitation", "rain", "snowfall",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "weather_code", "is_day",
]
_COMPARE_KEYS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
    "cloud_cover", "shortwave_radiation", "precipitation",
]


def _clip(payload: ObservationPayload) -> ObservationPayload:
    for k, (lo, hi) in PLAUSIBLE_RANGES.items():
        v = payload.get(k)
        if v is None:
            continue
        if lo is not None and v < lo:
            payload[k] = lo
        elif hi is not None and v > hi:
            payload[k] = hi
    return payload


def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _upsert(
    session: Session,
    location_id: int,
    payload: ObservationPayload,
    source: str,
    *,
    refresh: bool = False,
) -> bool:
    """Inserta la observación (alineada a la hora). Devuelve True si era nueva.

    ``refresh=True`` actualiza las medidas si la hora ya existe (para que la fila
    de la hora en curso refleje la última lectura de ``current``).
    """
    observed_at = _floor_hour(payload["observed_at"])
    measures = {k: payload.get(k) for k in _MEASURE_KEYS}
    values = {"location_id": location_id, "source": source, "observed_at": observed_at, **measures}
    stmt = pg_insert(WeatherObservation).values(values)
    conflict = ["location_id", "observed_at", "source"]
    if refresh:
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict,
            set_={**measures, "ingested_at": func.now()},
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict)
    result = session.execute(stmt)
    return bool(result.rowcount)


def backfill_recent(
    session: Session, provider: WeatherProvider, slug: str, past_days: int = 3
) -> int:
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")
    rows = provider.get_recent_hourly(past_days=past_days)
    n = sum(
        _upsert(session, loc.id, _clip(r), "realtime")
        for r in rows
        if r.get("observed_at")
    )
    log.info("Backfill: %d observaciones nuevas de las últimas %dd", n, past_days)
    return n


def ingest_current(session: Session, provider: WeatherProvider, slug: str) -> dict | None:
    """Descarga las condiciones actuales, las alinea a la hora, procesa y resume."""
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    payload = _clip(provider.get_current())
    payload["observed_at"] = _floor_hour(payload["observed_at"])

    # si hay un hueco de más de 1 h respecto a lo último almacenado, rellenarlo
    last = session.scalar(
        select(WeatherObservation.observed_at)
        .where(WeatherObservation.location_id == loc.id)
        .order_by(WeatherObservation.observed_at.desc())
        .limit(1)
    )
    if last is not None and (payload["observed_at"] - last) > timedelta(hours=1, minutes=30):
        for r in provider.get_recent_hourly(past_days=1):
            if r.get("observed_at"):
                _upsert(session, loc.id, _clip(r), "realtime")

    _upsert(session, loc.id, payload, "realtime", refresh=True)
    session.flush()

    summary = _process(session, slug, loc.id, payload)
    bus.publish("observation", {
        "observed_at": payload["observed_at"].isoformat(),
        "temperature_2m": payload.get("temperature_2m"),
        "is_anomaly": summary["comparison"].get("is_anomaly"),
    })
    return summary


def _recent_observations_df(session: Session, slug: str, hours: int = 60) -> pd.DataFrame:
    df = load_observations_df(session, slug)
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    df = df[df["timestamp"] >= pd.Timestamp(cutoff)]
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _active_champions(session: Session) -> dict[int, int]:
    rows = session.execute(
        select(ModelRun.horizon_hours, ModelRun.id).where(
            ModelRun.is_active, ModelRun.model_type.notlike("baseline%")
        )
    ).all()
    return {int(h): int(i) for h, i in rows}


def _process(session: Session, slug: str, location_id: int, payload: ObservationPayload) -> dict:
    observed_at: datetime = payload["observed_at"]
    values = {k: payload.get(k) for k in _COMPARE_KEYS}

    comparison = compare(session, slug, observed_at, values)

    # persistir la anomalía de esta observación
    if comparison.get("anomaly_score") is not None:
        obs_id = session.scalar(select(WeatherObservation.id).where(
            WeatherObservation.location_id == location_id,
            WeatherObservation.observed_at == observed_at,
            WeatherObservation.source == "realtime",
        ))
        if obs_id is not None:
            session.execute(
                pg_insert(Anomaly)
                .values(
                    location_id=location_id, observation_id=obs_id, observed_at=observed_at,
                    anomaly_score=comparison["anomaly_score"],
                    is_anomaly=comparison["is_anomaly"], detector="isolation_forest",
                    features_used=["temperature_2m", "relative_humidity_2m",
                                   "surface_pressure", "wind_speed_10m"],
                )
                .on_conflict_do_nothing(index_elements=["observation_id", "detector"])
            )

    predictions: list[dict] = []
    predictor = get_predictor()
    if predictor is not None:
        recent = _recent_observations_df(session, slug)
        try:
            preds = predictor.predict_from_observations(recent)
        except ValueError as exc:
            log.warning("Sin predicción (%s)", exc)
            preds = []
        champions = _active_champions(session)
        for p in preds:
            predictions.append(p.as_dict())
            run_id = champions.get(p.horizon_hours)
            if run_id is None:
                continue
            session.execute(
                pg_insert(Prediction)
                .values(
                    location_id=location_id, model_run_id=run_id,
                    base_time=p.base_time.to_pydatetime(),
                    target_time=p.target_time.to_pydatetime(),
                    predicted_temperature=p.predicted_temperature,
                    lower_bound=p.lower_bound, upper_bound=p.upper_bound,
                )
                .on_conflict_do_nothing(index_elements=["model_run_id", "base_time"])
            )

    # interpretación con Claude solo si hay anomalía y no hay una reciente (coste acotado)
    from backend.app.services.claude import maybe_generate_on_event

    maybe_generate_on_event(session, slug, is_anomaly=bool(comparison.get("is_anomaly")))

    return {"observed_at": observed_at, "comparison": comparison, "predictions": predictions}
