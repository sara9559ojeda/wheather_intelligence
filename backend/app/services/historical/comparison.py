"""Comparación de una observación con el histórico (fase 1, §15).

Para una observación dada calcula:
  - percentil de cada variable **condicionado a la época del año y la hora**
    (comparar septiembre contra septiembres, no contra todo el año)
  - a qué régimen (cluster K-Means) se parece
  - su puntuación de anomalía (Isolation Forest sobre residuales)
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import Integer, and_, cast, extract, select
from sqlalchemy.orm import Session

from backend.app.db.models import ClusterModel, WeatherObservation
from backend.app.db.queries import get_location
from backend.app.services.artifacts import get_anomaly_artifact, get_cluster_artifact

log = logging.getLogger(__name__)

COMPARE_VARS = ("temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m")


def _month_window(m: int) -> list[int]:
    return sorted({((m - 2) % 12) + 1, m, (m % 12) + 1})


def _hour_window(h: int) -> list[int]:
    return sorted({(h - 1) % 24, h, (h + 1) % 24})


def conditioned_percentiles(
    session: Session, location_id: int, observed_at: datetime, values: dict[str, float | None]
) -> list[dict]:
    months = _month_window(observed_at.month)
    hours = _hour_window(observed_at.hour)

    cols = [getattr(WeatherObservation, v) for v in COMPARE_VARS]
    stmt = (
        select(*cols)
        .where(
            and_(
                WeatherObservation.location_id == location_id,
                WeatherObservation.source == "historical",
                cast(extract("month", WeatherObservation.observed_at), Integer).in_(months),
                cast(extract("hour", WeatherObservation.observed_at), Integer).in_(hours),
            )
        )
    )
    df = pd.DataFrame(session.execute(stmt).all(), columns=list(COMPARE_VARS)).astype("float64")

    out: list[dict] = []
    for v in COMPARE_VARS:
        obs = values.get(v)
        series = df[v].dropna()
        if obs is None or series.empty:
            continue
        pct = round(100.0 * float((series < obs).mean()), 1)
        out.append({
            "variable": v,
            "value": round(float(obs), 2),
            "percentile": pct,
            "normal_range": (
                round(float(series.quantile(0.10)), 2),
                round(float(series.quantile(0.90)), 2),
            ),
        })
    return out


def _observation_frame(observed_at: datetime, values: dict) -> pd.DataFrame:
    ts = pd.Timestamp(observed_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    row = {"timestamp": ts}
    row.update({k: values.get(k) for k in (
        "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "cloud_cover", "shortwave_radiation", "precipitation",
    )})
    return pd.DataFrame([row])


def assign_cluster(observed_at: datetime, values: dict) -> dict | None:
    art = get_cluster_artifact()
    if art is None:
        return None
    from backend.ml.mining.clustering import ClusterModel

    model = ClusterModel(scaler=art["scaler"], kmeans=art["kmeans"],
                         features=art["features"], k=art["k"])
    df = _observation_frame(observed_at, values)
    try:
        labels, dist = model.assign(df)
    except Exception as exc:  # datos incompletos
        log.warning("No se pudo asignar cluster: %s", exc)
        return None
    if np.isnan(dist[0]):
        return None
    cid = int(labels[0])
    return {"cluster_id": cid, "distance_to_centroid": round(float(dist[0]), 3)}


def score_anomaly(observed_at: datetime, values: dict) -> dict | None:
    art = get_anomaly_artifact()
    if art is None:
        return None
    from backend.ml.mining.anomalies import AnomalyModel

    model = AnomalyModel(
        climatology=art["climatology"], iforest=art["iforest"],
        contamination=art["contamination"],
        resid_columns=[f"{v}_resid" for v in art["variables"]],
    )
    df = _observation_frame(observed_at, values)
    scored = model.score(df).iloc[0]
    if pd.isna(scored["anomaly_score"]):
        return None
    return {
        "anomaly_score": round(float(scored["anomaly_score"]), 5),
        "is_anomaly": bool(scored["is_anomaly"]),
    }


def compare(session: Session, slug: str, observed_at: datetime, values: dict) -> dict:
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    pcts = conditioned_percentiles(session, loc.id, observed_at, values)
    cluster = assign_cluster(observed_at, values)
    anomaly = score_anomaly(observed_at, values)

    cluster_label = None
    if cluster is not None:
        cluster_label = _cluster_labels(session).get(cluster["cluster_id"])

    return {
        "location": loc.name,
        "reference": "mismo mes ±1 y misma hora ±1, histórico completo",
        "observed_at": observed_at,
        "percentiles": pcts,
        "cluster_id": cluster["cluster_id"] if cluster else None,
        "cluster_label": cluster_label,
        "distance_to_centroid": cluster["distance_to_centroid"] if cluster else None,
        "is_anomaly": anomaly["is_anomaly"] if anomaly else None,
        "anomaly_score": anomaly["anomaly_score"] if anomaly else None,
    }


def _cluster_labels(session: Session) -> dict[int, str]:
    cm = session.scalar(
        select(ClusterModel).where(ClusterModel.is_active).order_by(ClusterModel.id.desc())
    )
    if cm is None or not cm.cluster_labels:
        return {}
    return {int(k): v for k, v in cm.cluster_labels.items()}
