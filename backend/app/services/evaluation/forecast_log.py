"""Registro de pronósticos emitidos y su verificación (evaluación transparente).

Reglas que protegen la validez de la evaluación:

1. **Solo se registra lo que se emite.** Cada pronóstico se guarda en el momento de
   emitirse, con el modelo y la versión que lo produjo. Nunca se reconstruyen
   pronósticos pasados con modelos nuevos.
2. **Idempotencia.** ``(ubicación, horizonte, hora base)`` es único: repetir la
   ingesta en la misma hora no duplica ni sobrescribe (se conserva la primera emisión).
3. **Sin fuga de futuro.** Se verifica solo cuando la hora objetivo ya pasó (con
   ``GRACE`` de margen). Como «observación» se acepta una fila ``realtime`` solo si
   se escribió DESPUÉS de empezar esa hora (``ingested_at >= target_time``): si no,
   podría ser el valor previsto que se guardó por adelantado al rellenar huecos.
4. **Referencias justas.** La persistencia («seguirá como ahora») y el pronóstico de
   Open-Meteo se capturan en el mismo instante de la emisión, no después.
5. **Las métricas solo usan filas verificadas.**

Limitación declarada: la «temperatura observada» es el dato que el sistema recibe de
Open-Meteo (análisis/nowcast o reanálisis ERA5), no una medición de estación.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.db.models import (
    FORECAST_EXPIRED,
    FORECAST_PENDING,
    FORECAST_VERIFIED,
    ForecastLog,
    Location,
    ModelRun,
    WeatherObservation,
)
from backend.app.services.prediction.predictor import HorizonPrediction

log = logging.getLogger(__name__)

GRACE = timedelta(minutes=90)           # espera tras la hora objetivo antes de verificar
EXPIRE_AFTER = timedelta(days=14)       # sin observación utilizable pasado este tiempo → expired
MIN_RELIABLE_N = 30                      # por debajo, la muestra es «pequeña» y se avisa
REFERENCE_SOURCE = "open_meteo_forecast"
HORIZONS = (1, 3, 6, 12, 24)


def _f(v) -> float | None:
    return None if v is None else float(v)


# ---------------------------------------------------------------- escritura
def record_forecasts(
    session: Session,
    location_id: int,
    preds: Sequence[HorizonPrediction],
    *,
    persistence_temp: float | None,
    reference: Mapping[datetime, float] | None,
    champions: Mapping[int, int],
) -> int:
    """Registra los pronósticos emitidos. Devuelve cuántos son nuevos (idempotente)."""
    new = 0
    for p in preds:
        target = p.target_time.to_pydatetime()
        ref = reference.get(target) if reference else None
        values = {
            "location_id": location_id, "horizon_hours": p.horizon_hours,
            "base_time": p.base_time.to_pydatetime(), "target_time": target,
            "predicted_temperature": p.predicted_temperature,
            "lower_bound": p.lower_bound, "upper_bound": p.upper_bound,
            "interval_level": p.interval_level, "interval_method": p.interval_method,
            "model_type": p.model_type, "model_run_id": champions.get(p.horizon_hours),
            "model_version": p.model_version,
            "baseline_persistence": persistence_temp,
            "reference_forecast": ref,
            "reference_source": REFERENCE_SOURCE if ref is not None else None,
            "status": FORECAST_PENDING,
        }
        # RETURNING distingue lo insertado de lo omitido (rowcount no es fiable con DO NOTHING)
        inserted = session.execute(
            pg_insert(ForecastLog).values(values).on_conflict_do_nothing(
                index_elements=["location_id", "horizon_hours", "base_time"]
            ).returning(ForecastLog.id)
        ).first()
        new += int(inserted is not None)
    return new


def missing_for_base_time(
    session: Session, location_id: int, base_time: datetime, horizons: Sequence[int]
) -> bool:
    """¿Falta registrar algún horizonte de esta hora base? (evita pedir la referencia de más)."""
    have = session.scalar(
        select(func.count()).select_from(ForecastLog).where(
            ForecastLog.location_id == location_id, ForecastLog.base_time == base_time,
            ForecastLog.horizon_hours.in_(list(horizons)),
        )
    )
    return (have or 0) < len(horizons)


def verify_pending(session: Session, *, now: datetime | None = None) -> dict[int, int]:
    """Verifica los pronósticos cuya hora objetivo ya pasó. Devuelve {location_id: n}.

    Idempotente: solo toca filas ``pending``; una fila verificada no se vuelve a tocar.
    No hace commit.
    """
    now = now or datetime.now(UTC)
    pending = session.scalars(
        select(ForecastLog)
        .where(ForecastLog.status == FORECAST_PENDING, ForecastLog.target_time <= now - GRACE)
        .order_by(ForecastLog.target_time).limit(5000)
    ).all()
    if not pending:
        return {}

    verified: dict[int, int] = {}
    by_loc: dict[int, list[ForecastLog]] = {}
    for row in pending:
        by_loc.setdefault(row.location_id, []).append(row)

    for loc_id, rows in by_loc.items():
        times = sorted({r.target_time for r in rows})
        obs = session.execute(
            select(WeatherObservation.observed_at, WeatherObservation.source,
                   WeatherObservation.temperature_2m, WeatherObservation.ingested_at)
            .where(WeatherObservation.location_id == loc_id,
                   WeatherObservation.observed_at.in_(times),
                   WeatherObservation.temperature_2m.is_not(None))
        ).all()
        index: dict[tuple[datetime, str], tuple[float, datetime]] = {
            (o.observed_at, o.source): (float(o.temperature_2m), o.ingested_at) for o in obs
        }
        for r in rows:
            chosen: tuple[float, str] | None = None
            rt = index.get((r.target_time, "realtime"))
            if rt is not None and rt[1] >= r.target_time:        # medido durante/después de la hora
                chosen = (rt[0], "realtime")
            elif (r.target_time, "historical") in index:
                chosen = (index[(r.target_time, "historical")][0], "historical")

            if chosen is not None:
                r.status = FORECAST_VERIFIED
                r.observed_temperature = chosen[0]
                r.observed_source = chosen[1]
                r.verified_at = now
                r.error = round(float(r.predicted_temperature) - chosen[0], 3)
                verified[loc_id] = verified.get(loc_id, 0) + 1
            elif r.target_time < now - EXPIRE_AFTER:
                r.status = FORECAST_EXPIRED
    session.flush()
    return verified


# ---------------------------------------------------------------- lectura / métricas
def verified_frame(session: Session, location_id: int, days: int,
                   now: datetime | None = None) -> pd.DataFrame:
    """Pronósticos verificados con hora objetivo dentro de los últimos ``days`` días."""
    now = now or datetime.now(UTC)
    rows = session.execute(
        select(
            ForecastLog.horizon_hours, ForecastLog.target_time, ForecastLog.predicted_temperature,
            ForecastLog.lower_bound, ForecastLog.upper_bound, ForecastLog.observed_temperature,
            ForecastLog.baseline_persistence, ForecastLog.reference_forecast,
            ForecastLog.model_type, ForecastLog.model_version, ForecastLog.verified_at,
            ForecastLog.observed_source,
        ).where(
            ForecastLog.location_id == location_id, ForecastLog.status == FORECAST_VERIFIED,
            ForecastLog.target_time >= now - timedelta(days=days),
        ).order_by(ForecastLog.target_time)
    ).all()
    cols = ["horizon", "target_time", "predicted", "lower", "upper", "observed", "persistence",
            "reference", "model_type", "model_version", "verified_at", "observed_source"]
    df = pd.DataFrame(rows, columns=cols)
    num = ["predicted", "lower", "upper", "observed", "persistence", "reference"]
    df[num] = df[num].astype("float64")
    if not df.empty:
        df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    return df


def _mae_rmse(err: np.ndarray) -> tuple[float, float]:
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def _fmt(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def horizon_metrics(df: pd.DataFrame, horizon: int, days: int) -> dict:
    """Métricas de un horizonte a partir de pronósticos YA verificados."""
    sub = df[df["horizon"] == horizon] if not df.empty else df
    n = len(sub)
    base = {"horizon_hours": horizon, "n_verified": n, "model_versions": [], "model_types": []}
    if n == 0:
        return {**base, "explanation": (
            f"Todavía no hay pronósticos verificados a {horizon} h en los últimos {days} días.")}

    err = (sub["predicted"] - sub["observed"]).to_numpy()
    mae, rmse = _mae_rmse(err)
    out = {
        **base, "mae": round(mae, 3), "rmse": round(rmse, 3), "bias": round(float(err.mean()), 3),
        "model_versions": sorted({v for v in sub["model_version"].dropna()}),
        "model_types": sorted(set(sub["model_type"])),
        "last_verified_at": sub["verified_at"].max().to_pydatetime(),
    }

    pers = sub.dropna(subset=["persistence"])
    if len(pers):
        pe = (pers["persistence"] - pers["observed"]).to_numpy()
        pmae, prmse = _mae_rmse(pe)
        _, m_rmse = _mae_rmse((pers["predicted"] - pers["observed"]).to_numpy())
        out["persistence"] = {"n": len(pers), "mae": round(pmae, 3), "rmse": round(prmse, 3)}
        out["skill_vs_persistence"] = round(1 - m_rmse / prmse, 3) if prmse > 0 else None

    ref = sub.dropna(subset=["reference"])
    if len(ref):
        re_ = (ref["reference"] - ref["observed"]).to_numpy()
        rmae, rrmse = _mae_rmse(re_)
        mm, mr = _mae_rmse((ref["predicted"] - ref["observed"]).to_numpy())
        out["reference"] = {"n": len(ref), "mae": round(rmae, 3), "rmse": round(rrmse, 3)}
        out["model_on_reference_rows"] = {"n": len(ref), "mae": round(mm, 3), "rmse": round(mr, 3)}

    bounded = sub.dropna(subset=["lower", "upper"])
    if len(bounded):
        inside = bounded["observed"].between(bounded["lower"], bounded["upper"])
        out["coverage_pct"] = round(float(inside.mean() * 100), 1)
        out["mean_interval_width"] = round(float((bounded["upper"] - bounded["lower"]).mean()), 2)

    text = (f"En los últimos {days} días se verificaron {n} pronósticos a {horizon} h: el error "
            f"absoluto promedio fue de {_fmt(mae)} °C (RMSE {_fmt(rmse)} °C; sesgo "
            f"{_fmt(float(err.mean()))} °C, positivo = el modelo sobreestima).")
    if "persistence" in out:
        pmae = _fmt(out["persistence"]["mae"])
        text += f" Suponer que la temperatura no cambia habría errado {pmae} °C de media."
        sk = out.get("skill_vs_persistence")
        if sk is not None:
            text += (f" El modelo mejora a esa referencia un {_fmt(sk * 100, 0)} % (RMSE)."
                     if sk > 0 else " En este periodo el modelo NO supera a esa referencia.")
    if "reference" in out:
        ref_mae = _fmt(out["reference"]["mae"])
        mod_mae = _fmt(out["model_on_reference_rows"]["mae"])
        text += (f" En las mismas {out['reference']['n']} horas, el pronóstico de Open-Meteo "
                 f"erró {ref_mae} °C frente a {mod_mae} °C del modelo.")
    if n < MIN_RELIABLE_N:
        text += f" Ojo: son solo {n} casos; con tan pocos las cifras son orientativas."
    out["explanation"] = text
    return out


def daily_series(df: pd.DataFrame, horizon: int) -> list[dict]:
    sub = df[df["horizon"] == horizon] if not df.empty else df
    if sub.empty:
        return []
    err = (sub["predicted"] - sub["observed"]).abs()
    tmp = pd.DataFrame({"day": sub["target_time"].dt.strftime("%Y-%m-%d"),
                        "ae": err.to_numpy(), "se": (err**2).to_numpy()})
    g = tmp.groupby("day").agg(n=("ae", "size"), mae=("ae", "mean"), mse=("se", "mean"))
    return [{"day": d, "n": int(r.n), "mae": round(float(r.mae), 3),
             "rmse": round(float(np.sqrt(r.mse)), 3)} for d, r in g.iterrows()]


def offline_models(session: Session, location_id: int) -> list[dict]:
    """Candidatos del último entrenamiento: métricas de VALIDACIÓN (no son pronósticos reales)."""
    runs = session.scalars(
        select(ModelRun).where(ModelRun.location_id == location_id)
        .order_by(ModelRun.trained_at.desc(), ModelRun.id.desc())
    ).all()
    seen: set[tuple[int, str]] = set()
    out: list[dict] = []
    for r in runs:
        key = (r.horizon_hours, r.model_type)
        if key in seen or not (r.metrics or {}).get("validation"):
            continue
        seen.add(key)
        test = (r.metrics or {}).get("test") or {}
        out.append({
            "horizon_hours": r.horizon_hours, "model_type": r.model_type,
            "is_champion": bool(r.is_active),
            "valid_rmse": _f(r.rmse), "valid_mae": _f(r.mae),
            "test_rmse": test.get("rmse"), "test_mae": test.get("mae"),
            "trained_at": r.trained_at,
        })
    return sorted(out, key=lambda m: (m["horizon_hours"], m["valid_rmse"] or 99))


def evaluation_summary(session: Session, loc: Location, days: int) -> dict:
    now = datetime.now(UTC)
    totals = session.execute(
        select(
            func.count(), func.count().filter(ForecastLog.status == FORECAST_VERIFIED),
            func.count().filter(ForecastLog.status == FORECAST_PENDING),
            func.min(ForecastLog.issued_at), func.max(ForecastLog.verified_at),
        ).where(ForecastLog.location_id == loc.id)
    ).one()
    n_logged, n_verified, n_pending, first_logged, last_verified = totals
    df = verified_frame(session, loc.id, days, now)

    notes = [
        "Solo cuentan pronósticos emitidos antes de conocer el valor observado.",
        "«Observado» es el dato recibido de Open-Meteo (análisis/reanálisis), no una estación.",
        "El error medido en el pasado no garantiza la precisión de pronósticos futuros.",
    ]
    status = "ok" if n_verified else ("collecting" if n_logged else "unavailable")
    if status == "collecting":
        notes.insert(0, "El registro de pronósticos acaba de empezar: aún no hay ninguno "
                        "verificado. Las métricas aparecerán cuando pasen sus horas objetivo.")
    elif status == "unavailable":
        notes.insert(0, "Aún no se ha emitido ningún pronóstico registrado para esta ciudad.")

    horizons_present = sorted({int(h) for h in df["horizon"]}) if not df.empty else []
    horizons = sorted(set(HORIZONS) | set(horizons_present))
    return {
        "status": status, "location": loc.name, "slug": loc.slug, "window_days": days,
        "generated_at": now, "n_logged_total": int(n_logged), "n_verified_total": int(n_verified),
        "n_pending": int(n_pending), "first_logged_at": first_logged,
        "last_verified_at": last_verified,
        "horizons": [horizon_metrics(df, h, days) for h in horizons],
        "series": [{"horizon_hours": h, "points": daily_series(df, h)} for h in horizons_present],
        "offline_models": offline_models(session, loc.id),
        "notes": notes,
    }


def verified_pairs(session: Session, loc: Location, days: int, horizon: int,
                   limit: int = 500) -> dict:
    df = verified_frame(session, loc.id, days)
    sub = df[df["horizon"] == horizon] if not df.empty else df
    total = len(sub)
    sub = sub.tail(limit)
    pts = [{"target_time": r.target_time.to_pydatetime(), "predicted": r.predicted,
            "observed": r.observed,
            "lower": None if pd.isna(r.lower) else r.lower,
            "upper": None if pd.isna(r.upper) else r.upper} for r in sub.itertuples()]
    return {"horizon_hours": horizon, "window_days": days, "n_total": total, "points": pts}
