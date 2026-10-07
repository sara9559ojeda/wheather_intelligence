"""Endpoints de estadística, anomalías y clusters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_location_slug
from backend.app.db.models import Anomaly, ClusterModel, WeatherObservation
from backend.app.db.queries import get_location
from backend.app.schemas import AnomalyOut
from backend.app.schemas.models import ClusterModelOut, ClusterOut, StatsOut, StatVariable

router = APIRouter(tags=["analytics"])

_STAT_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
              "wind_speed_10m", "cloud_cover", "shortwave_radiation"]


@router.get("/stats", response_model=StatsOut)
def stats(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    days: int = Query(365, ge=1, le=366 * 30),
) -> StatsOut:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    cols = [getattr(WeatherObservation, v) for v in _STAT_VARS]
    rows = db.execute(
        select(*cols).where(
            WeatherObservation.location_id == loc.id,
            WeatherObservation.observed_at >= cutoff,
        )
    ).all()
    df = pd.DataFrame(rows, columns=_STAT_VARS).astype("float64")
    variables = []
    for v in _STAT_VARS:
        s = df[v].dropna()
        if s.empty:
            continue
        variables.append(StatVariable(
            variable=v, mean=round(s.mean(), 3), median=round(s.median(), 3),
            std=round(s.std(), 3), min=round(s.min(), 3),
            p05=round(s.quantile(0.05), 3), p25=round(s.quantile(0.25), 3),
            p75=round(s.quantile(0.75), 3), p95=round(s.quantile(0.95), 3),
            max=round(s.max(), 3),
        ))
    return StatsOut(location=loc.name, period=f"últimos {days} días",
                    n_rows=len(df), variables=variables)


@router.get("/stats/correlations")
def correlations(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    days: int = Query(120, ge=7, le=366 * 30),
) -> dict:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    cols = [getattr(WeatherObservation, v) for v in _STAT_VARS]
    rows = db.execute(
        select(*cols).where(
            WeatherObservation.location_id == loc.id,
            WeatherObservation.observed_at >= cutoff,
        )
    ).all()
    df = pd.DataFrame(rows, columns=_STAT_VARS).astype("float64")
    corr = df.corr(method="pearson").round(3)
    return {
        "variables": _STAT_VARS,
        "matrix": corr.where(pd.notna(corr), None).values.tolist(),
        "method": "pearson",
        "n_rows": int(len(df)),
        "period": f"últimos {days} días",
    }


@router.get("/anomalies", response_model=list[AnomalyOut])
def anomalies(
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    limit: int = Query(50, ge=1, le=500),
    only_flagged: bool = Query(True),
) -> list[AnomalyOut]:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    stmt = (
        select(Anomaly, WeatherObservation)
        .join(WeatherObservation, WeatherObservation.id == Anomaly.observation_id)
        .where(Anomaly.location_id == loc.id)
        .order_by(Anomaly.observed_at.desc())
        .limit(limit)
    )
    if only_flagged:
        stmt = stmt.where(Anomaly.is_anomaly.is_(True))
    out = []
    for anomaly, obs in db.execute(stmt).all():
        out.append(AnomalyOut(
            observed_at=anomaly.observed_at,
            anomaly_score=float(anomaly.anomaly_score),
            is_anomaly=anomaly.is_anomaly,
            detector=anomaly.detector,
            temperature_2m=_f(obs.temperature_2m),
            relative_humidity_2m=_f(obs.relative_humidity_2m),
            surface_pressure=_f(obs.surface_pressure),
            wind_speed_10m=_f(obs.wind_speed_10m),
        ))
    return out


@router.get("/clusters", response_model=ClusterModelOut)
def clusters(
    db: Session = Depends(get_db), slug: str = Depends(get_location_slug)
) -> ClusterModelOut:
    loc = get_location(db, slug)
    if loc is None:
        raise HTTPException(404, "Ubicación no encontrada")
    cm = db.scalar(
        select(ClusterModel).where(ClusterModel.is_active).order_by(ClusterModel.id.desc())
    )
    if cm is None:
        raise HTTPException(503, "Todavía no hay un modelo de clustering")

    centroids = {int(row["cluster"]): row for row in (cm.centroids or [])}
    labels = {int(k): v for k, v in (cm.cluster_labels or {}).items()}
    clusters_out = []
    for cid in sorted(labels):
        row = centroids.get(cid, {})
        clusters_out.append(ClusterOut(
            cluster_id=cid,
            label=labels[cid],
            share_pct=float(row.get("pct", 0.0)),
            centroid={k: float(v) for k, v in row.items()
                      if k in _STAT_VARS or k == "precipitation"},
        ))
    return ClusterModelOut(
        k=cm.k, silhouette=_f(cm.silhouette), trained_at=cm.trained_at, clusters=clusters_out
    )


def _f(v) -> float | None:
    return float(v) if v is not None else None
