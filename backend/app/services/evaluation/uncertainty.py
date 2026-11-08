"""Incertidumbre explicable: calibración de intervalos, cobertura y confiabilidad.

Estado de los intervalos
------------------------
Los modelos entrenados traen un intervalo del 90 % construido con los percentiles
5 y 95 de los **residuales de validación** (un conformal «split» sobre un tramo
temporal posterior al de entrenamiento). Eso es correcto si el error futuro se
parece al de validación, pero el tiempo cambia: por eso se mide su cobertura real.

Calibración en vivo (conformal con ventana móvil)
-------------------------------------------------
Cuando hay suficientes pronósticos YA verificados del modelo vigente
(``MIN_CALIBRATION``), el intervalo se recentra y reescala con los cuantiles
(con la corrección de muestra finita de conformal) de sus residuales más
recientes. Solo se usan errores ya conocidos al emitir (sin fuga de futuro).
Supuestos y límites: los errores meteorológicos están autocorrelacionados, así
que la garantía de cobertura de conformal es aproximada y no condicional; una
ventana móvil permite adaptarse a cambios de régimen a costa de más ruido.

Confiabilidad
-------------
La etiqueta NO sale del ancho del intervalo, sino de métricas verificadas
(ver ``CRITERIA``). Con pocos casos se dice «preliminar» o «no evaluable».
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import FORECAST_VERIFIED, ForecastLog, Location, ModelRun
from backend.app.services.evaluation.forecast_log import MIN_RELIABLE_N, verified_frame
from backend.app.services.prediction.predictor import NOMINAL_LEVEL, HorizonPrediction

MIN_CALIBRATION = 100        # pronósticos verificados necesarios para recalibrar
CALIBRATION_WINDOW = 300     # los más recientes
METHOD_STATIC = "validation_residual_quantiles"
METHOD_ROLLING = "conformal_rolling"
EVAL_DAYS = 30
RECENT_DAYS = 7
DEGRADATION_RATIO = 1.25     # MAE reciente / MAE previo
DEGRADATION_VS_OFFLINE = 1.5  # MAE reciente / MAE del periodo de prueba

CRITERIA = [
    f"Con al menos {MIN_RELIABLE_N} pronósticos verificados del modelo vigente se compara la "
    "cobertura real con el nivel nominal (90 %).",
    "Alta: cobertura a menos de 5 puntos del nivel nominal, al menos 100 casos, sin deterioro "
    "y sin quedar por debajo de la persistencia.",
    "Media: cobertura a menos de 10 puntos del nominal (o «alta» con menos de 100 casos o "
    "con un solo problema).",
    "Baja: cobertura más de 10 puntos por debajo del nominal, o varios problemas a la vez.",
    "Deterioro: el error medio de los últimos 7 días supera en un 25 % al de los 23 días "
    "previos (o en un 50 % al del periodo de prueba).",
    "Preliminar: sin suficientes casos reales, solo se conoce la cobertura offline (periodo de "
    "prueba); no evaluable: ni siquiera eso.",
]


# ---------------------------------------------------------------- conformal
def conformal_bounds(residuals: np.ndarray, level: float = NOMINAL_LEVEL) -> tuple[float, float]:
    """Cuantiles inferior/superior de residuales (observado − predicho), conformal split.

    Con ``n`` residuales, el límite superior es el de rango ``ceil((n+1)(1−α/2))`` y el
    inferior el de rango ``floor((n+1)·α/2)``; así se corrige el sesgo de muestra finita.
    """
    res = np.sort(np.asarray(residuals, dtype="float64"))
    n = len(res)
    if n == 0:
        raise ValueError("Sin residuales")
    alpha = 1 - level
    hi_rank = min(n, math.ceil((n + 1) * (1 - alpha / 2)))
    lo_rank = max(1, math.floor((n + 1) * alpha / 2))
    return float(res[lo_rank - 1]), float(res[hi_rank - 1])


def recent_residuals(session: Session, location_id: int, horizon: int, model_version: str | None,
                     limit: int = CALIBRATION_WINDOW) -> np.ndarray:
    """Residuales (observado − predicho) YA conocidos del modelo vigente, los más recientes."""
    if not model_version:
        return np.array([])
    rows = session.scalars(
        select(ForecastLog.error).where(
            ForecastLog.location_id == location_id, ForecastLog.horizon_hours == horizon,
            ForecastLog.status == FORECAST_VERIFIED, ForecastLog.model_version == model_version,
            ForecastLog.error.is_not(None),
        ).order_by(ForecastLog.target_time.desc()).limit(limit)
    ).all()
    return -np.array([float(e) for e in rows])      # error = predicho − observado


def apply_calibration(
    session: Session, location_id: int, preds: list[HorizonPrediction]
) -> list[HorizonPrediction]:
    """Sustituye el intervalo estático por el conformal móvil cuando hay datos para ello."""
    for p in preds:
        p.interval_level = NOMINAL_LEVEL
        res = recent_residuals(session, location_id, p.horizon_hours, p.model_version)
        if len(res) < MIN_CALIBRATION:
            p.interval_method = METHOD_STATIC
            continue
        lo, hi = conformal_bounds(res)
        p.lower_bound = round(p.predicted_temperature + lo, 2)
        p.upper_bound = round(p.predicted_temperature + hi, 2)
        p.interval_method = METHOD_ROLLING
    return preds


# ---------------------------------------------------------------- confiabilidad
def classify_reliability(
    *, n_live: int, live_coverage: float | None, skill: float | None, degraded: bool,
    offline_coverage: float | None, nominal: float = NOMINAL_LEVEL,
) -> tuple[str, str, list[str]]:
    """(nivel, etiqueta, motivos). Criterios documentados en ``CRITERIA``."""
    if n_live < MIN_RELIABLE_N or live_coverage is None:
        if offline_coverage is not None:
            gap = nominal - offline_coverage
            reasons = [
                f"Solo hay {n_live} pronósticos verificados (mínimo {MIN_RELIABLE_N}).",
                f"En el periodo de prueba offline los intervalos cubrieron "
                f"{offline_coverage * 100:.0f} % de los casos (nominal {nominal * 100:.0f} %).",
            ]
            if gap > 0.05:
                reasons.append("Esa cobertura offline queda por debajo del nominal.")
            return "preliminar", "Preliminar", reasons
        return "no_evaluable", "No evaluable aún", [
            f"Solo hay {n_live} pronósticos verificados y no hay evaluación offline."]

    gap = nominal - live_coverage
    problems: list[str] = []
    reasons = [f"Cobertura real {live_coverage * 100:.0f} % con {n_live} casos "
               f"(nominal {nominal * 100:.0f} %)."]
    level = 0 if gap <= 0.05 else 1 if gap <= 0.10 else 2
    if gap > 0.05:
        problems.append("cobertura por debajo del nominal")
    if n_live < 100 and level == 0:
        level = 1
        reasons.append("Muestra moderada (menos de 100 casos): no se llega a «alta».")
    if skill is not None and skill < 0:
        problems.append("no supera a la persistencia")
        reasons.append("En este horizonte el modelo no mejora a «seguirá igual que ahora».")
    if degraded:
        problems.append("deterioro reciente")
        reasons.append("El error de la última semana es claramente mayor que el habitual.")
    if len(problems) >= 2 and level < 2:
        level += 1
    elif problems and level == 0:
        level = 1
    if live_coverage > nominal + 0.07:
        reasons.append("Los intervalos cubren de más: son conservadores (más anchos de lo "
                       "necesario).")
    name = ("alta", "media", "baja")[min(level, 2)]
    return name, name.capitalize(), reasons


def _offline_block(session: Session, location_id: int, horizon: int) -> dict | None:
    run = session.scalar(
        select(ModelRun).where(
            ModelRun.location_id == location_id, ModelRun.horizon_hours == horizon,
            ModelRun.is_active.is_(True), ModelRun.model_type.notlike("baseline%"),
        ).order_by(ModelRun.id.desc()).limit(1)
    )
    return (run.metrics or {}).get("interval_eval") if run else None


def _offline_test_mae(session: Session, location_id: int, horizon: int) -> float | None:
    run = session.scalar(
        select(ModelRun).where(
            ModelRun.location_id == location_id, ModelRun.horizon_hours == horizon,
            ModelRun.is_active.is_(True), ModelRun.model_type.notlike("baseline%"),
        ).order_by(ModelRun.id.desc()).limit(1)
    )
    test = (run.metrics or {}).get("test") if run else None
    return float(test["mae"]) if test and test.get("mae") is not None else None


def _fmt(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def horizon_reliability(session: Session, loc: Location, pred: HorizonPrediction,
                        verified, now: datetime) -> dict:
    h = pred.horizon_hours
    sub = verified[(verified["horizon"] == h) & (verified["model_version"] == pred.model_version)] \
        if not verified.empty else verified
    n_live = len(sub)
    live_cov = live_width = skill = None
    degraded = False
    if n_live:
        bounded = sub.dropna(subset=["lower", "upper"])
        if len(bounded):
            inside = bounded["observed"].between(bounded["lower"], bounded["upper"])
            live_cov = float(inside.mean())
            live_width = float((bounded["upper"] - bounded["lower"]).mean())
        pers = sub.dropna(subset=["persistence"])
        if len(pers):
            m = float(np.sqrt(np.mean((pers["predicted"] - pers["observed"]) ** 2)))
            b = float(np.sqrt(np.mean((pers["persistence"] - pers["observed"]) ** 2)))
            skill = 1 - m / b if b > 0 else None
        ae = (sub["predicted"] - sub["observed"]).abs()
        recent_mask = sub["target_time"] >= now - timedelta(days=RECENT_DAYS)
        n_recent, n_prev = int(recent_mask.sum()), int((~recent_mask).sum())
        if n_recent >= 20 and n_prev >= 20:
            recent_mae = float(ae[recent_mask].mean())
            degraded = recent_mae > DEGRADATION_RATIO * float(ae[~recent_mask].mean())
        elif n_recent >= 20:
            off_mae = _offline_test_mae(session, loc.id, h)
            recent_mae = float(ae[recent_mask].mean())
            degraded = bool(off_mae and recent_mae > DEGRADATION_VS_OFFLINE * off_mae)

    off = _offline_block(session, loc.id, h)
    off_cov = off.get("coverage_test") if off else None
    level, label, reasons = classify_reliability(
        n_live=n_live, live_coverage=live_cov, skill=skill, degraded=degraded,
        offline_coverage=off_cov, nominal=pred.interval_level or NOMINAL_LEVEL)

    nominal = pred.interval_level or NOMINAL_LEVEL
    statement = (f"Para dentro de {h} {'hora' if h == 1 else 'horas'}, el modelo estima "
                 f"{_fmt(pred.predicted_temperature)} °C y un intervalo predictivo de "
                 f"{_fmt(pred.lower_bound)} a {_fmt(pred.upper_bound)} °C "
                 f"(nivel nominal {nominal * 100:.0f} %).")
    if live_cov is not None and n_live >= MIN_RELIABLE_N:
        cov_text = (f"En evaluaciones recientes, los intervalos de este horizonte incluyeron el "
                    f"valor observado en el {live_cov * 100:.0f} % de los casos ({n_live} "
                    "pronósticos verificados). Es una cobertura histórica, no la probabilidad de "
                    "que este pronóstico concreto acierte.")
    elif n_live:
        cov_text = (f"Solo hay {n_live} pronósticos verificados de este horizonte (mínimo "
                    f"{MIN_RELIABLE_N}): todavía no se puede estimar la cobertura real.")
    else:
        cov_text = "Todavía no hay pronósticos verificados de este horizonte con el modelo vigente."
    if off_cov is not None and n_live < MIN_RELIABLE_N:
        cov_text += (f" En el periodo de prueba histórico (offline) los intervalos cubrieron el "
                     f"{off_cov * 100:.0f} % de {off['n_test']} casos.")

    return {
        "horizon_hours": h, "predicted_temperature": pred.predicted_temperature,
        "lower_bound": pred.lower_bound, "upper_bound": pred.upper_bound,
        "interval_width": round(pred.upper_bound - pred.lower_bound, 2),
        "nominal_level": nominal, "interval_method": pred.interval_method,
        "base_time": pred.base_time.to_pydatetime(),
        "target_time": pred.target_time.to_pydatetime(),
        "live_coverage_pct": None if live_cov is None else round(live_cov * 100, 1),
        "live_n": n_live, "live_mean_width": None if live_width is None else round(live_width, 2),
        "offline_coverage_pct": None if off_cov is None else round(off_cov * 100, 1),
        "offline_n": off.get("n_test") if off else None,
        "offline_mean_width": off.get("mean_width_test") if off else None,
        "level": level, "level_label": label, "degraded": degraded, "reasons": reasons,
        "statement": statement, "coverage_statement": cov_text,
    }


def reliability_report(session: Session, loc: Location, preds: list[HorizonPrediction]) -> dict:
    now = datetime.now(UTC)
    base = {"location": loc.name, "slug": loc.slug, "generated_at": now, "criteria": CRITERIA,
            "notes": [
                "La cobertura es una frecuencia histórica de aciertos del intervalo, no la "
                "probabilidad individual de un pronóstico.",
                "Los intervalos se recalibran con conformal móvil al reunir "
                f"{MIN_CALIBRATION} pronósticos verificados del modelo vigente.",
            ]}
    if not preds:
        return {**base, "status": "unavailable", "horizons": []}
    verified = verified_frame(session, loc.id, EVAL_DAYS, now)
    return {**base, "status": "ok", "horizons": [
        horizon_reliability(session, loc, p, verified, now) for p in sorted(
            preds, key=lambda x: x.horizon_hours)]}
