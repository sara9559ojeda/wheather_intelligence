"""Contexto meteorológico estructurado que recibe el analista.

Reúne, para UNA ciudad, todo lo que el sistema sabe (observación actual,
normalidad histórica, régimen, pronósticos con su incertidumbre, evaluación de
los pronósticos emitidos y modelos) y declara qué partes están disponibles.

Es la **única** fuente de hechos para el analista, con Claude o sin él: Claude
recibe este JSON y tiene prohibido salirse de él. No contiene credenciales,
tokens ni datos de usuarios.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.app.db.queries import get_location, latest_observation
from backend.app.services.analysis.normality import analyze_normality
from backend.app.services.analysis.regimes import analyze_regimes
from backend.app.services.evaluation.forecast_log import evaluation_summary, offline_models
from backend.app.services.evaluation.uncertainty import reliability_report
from backend.app.services.prediction.live import live_forecasts

log = logging.getLogger(__name__)

CONTEXT_TTL_SECONDS = 60
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _r(v, d: int = 1):
    return None if v is None else round(float(v), d)


def _current(session: Session, location_id: int) -> dict | None:
    obs = latest_observation(session, location_id)
    if obs is None:
        return None
    age = max(0.0, (datetime.now(UTC) - obs.observed_at).total_seconds() / 60)
    return {
        "observado_utc": obs.observed_at.isoformat(), "antiguedad_min": round(age),
        "retrasado": age > 120, "fuente": obs.source,
        "temperatura_c": _r(obs.temperature_2m), "humedad_pct": _r(obs.relative_humidity_2m, 0),
        "viento_kmh": _r(obs.wind_speed_10m), "precipitacion_mm": _r(obs.precipitation),
        "presion_hpa": _r(obs.surface_pressure, 0), "nubosidad_pct": _r(obs.cloud_cover, 0),
    }


def _normality(n: dict) -> dict | None:
    if n.get("status") not in ("ok", "partial"):
        return None
    ref = n["reference"]
    an = n.get("anomaly") or {}
    sim = n.get("similar") or {}
    return {
        "resumen": n["summary"],
        "referencia": {
            "descripcion": ref["description"], "n_observaciones": ref["n_observations"],
            "anios": ref["years"], "periodo_inicio": ref["period_start"].date().isoformat(),
            "periodo_fin": ref["period_end"].date().isoformat(),
            "representatividad": ref["representativeness"], "ventana_ampliada": ref["widened"],
        },
        "variables": [
            {"variable": v["variable"], "nombre": v["label"], "unidad": v["unit"],
             "valor": v.get("value"), "percentil": v.get("percentile"),
             "clasificacion": v.get("classification_label"), "extremo_univariable": v.get("is_extreme", False),
             "rango_habitual_p10_p90": v.get("usual_range"), "mediana": v.get("median"),
             "estado": v["status"]}
            for v in n["variables"]
        ],
        "anomalia": {
            "disponible": an.get("available", False), "tipo": an.get("kind"),
            "multivariable": an.get("is_multivariate_anomaly"), "puntuacion": an.get("score"),
            "variables_mas_alejadas": [
                {"nombre": c["label"], "percentil": c["percentile"],
                 "direccion": "por encima" if c["direction"] == "above" else "por debajo"}
                for c in an.get("contributors", [])],
            "explicacion": an.get("explanation"),
        },
        "situaciones_parecidas": {
            "disponible": sim.get("available", False), "porcentaje": sim.get("share_pct"),
            "anios_con_casos": sim.get("years_with_similar"), "anios_totales": sim.get("years_total"),
        },
    }


def _regime(r: dict) -> dict | None:
    cur = r.get("current")
    if r.get("status") not in ("ok", "partial") or not cur:
        return None
    prof = next(p for p in r["regimes"] if p["regime_id"] == cur["regime_id"])
    pers = prof.get("persistence")
    return {
        "actual": {
            "etiqueta": prof["label"], "descripcion": prof["description"],
            "frecuencia_historica_pct": prof["share_pct"],
            "distancia_al_centroide": cur["distance_to_centroid"], "distancia": cur["distance_band"],
            "permanencia_mediana_h": pers["median_hours"] if pers else None,
            "meses_pico": prof["peak_months"],
            "variables_dominantes": [
                {"nombre": d["label"], "valor_tipico": d["median"], "unidad": d["unit"],
                 "direccion": "por encima" if d["direction"] == "above" else "por debajo"}
                for d in prof["dominant"]],
            "segundo_regimen_cercano": cur.get("second_label"),
            "razon_distancia_segundo": cur.get("distance_ratio_to_second"),
        },
        "transiciones_historicas": [
            {"hacia": t["label"], "frecuencia": round(t["probability"], 3), "casos": t["count"]}
            for t in r.get("transitions_from_current", [])] if r.get("transitions_supported")
        else [],
        "regimenes_de_la_ciudad": [
            {"etiqueta": p["label"], "frecuencia_pct": p["share_pct"]} for p in r["regimes"]],
        "compuesto_ultimos_7_dias_pct": {
            next(p["label"] for p in r["regimes"] if p["regime_id"] == k): v
            for k, v in r.get("recent_share_pct", {}).items()},
        "nota": "K-Means asigna al centroide más cercano; no son probabilidades de pertenencia.",
    }


def _forecast(rep: dict) -> list[dict] | None:
    if rep.get("status") != "ok":
        return None
    return [
        {"horizonte_h": h["horizon_hours"], "temperatura_c": h["predicted_temperature"],
         "minimo_c": h["lower_bound"], "maximo_c": h["upper_bound"],
         "ancho_c": h["interval_width"], "nivel_nominal": h["nominal_level"],
         "metodo_intervalo": h["interval_method"], "confiabilidad": h["level"],
         "confiabilidad_motivos": h["reasons"], "cobertura_real_pct": h["live_coverage_pct"],
         "pronosticos_verificados": h["live_n"], "cobertura_offline_pct": h["offline_coverage_pct"],
         "deterioro_reciente": h["degraded"], "objetivo_utc": h["target_time"].isoformat()}
        for h in rep["horizons"]
    ]


def _evaluation(ev: dict) -> dict | None:
    rows = [h for h in ev["horizons"] if h["n_verified"]]
    if not rows:
        return None
    return {
        "ventana_dias": ev["window_days"], "pronosticos_verificados_total": ev["n_verified_total"],
        "horizontes": [
            {"horizonte_h": h["horizon_hours"], "n": h["n_verified"], "mae_c": h["mae"],
             "rmse_c": h["rmse"], "sesgo_c": h["bias"],
             "mae_persistencia_c": (h.get("persistence") or {}).get("mae"),
             "mejora_vs_persistencia": h.get("skill_vs_persistence"),
             "mae_open_meteo_c": (h.get("reference") or {}).get("mae"),
             "mae_modelo_en_esas_horas_c": (h.get("model_on_reference_rows") or {}).get("mae")}
            for h in rows],
        "nota": "«Observado» = dato de Open-Meteo, no estación. Error pasado ≠ precisión futura.",
    }


def _models(session: Session, location_id: int) -> list[dict] | None:
    rows = offline_models(session, location_id)
    if not rows:
        return None
    return [
        {"horizonte_h": m["horizon_hours"], "modelo": m["model_type"], "campeon": m["is_champion"],
         "rmse_validacion_c": _r(m["valid_rmse"], 3), "rmse_prueba_c": _r(m["test_rmse"], 3),
         "mae_prueba_c": _r(m["test_mae"], 3)}
        for m in rows if m["is_champion"] or m["model_type"].startswith("baseline")
        or m["valid_rmse"] is not None
    ]


def build_context(session: Session, slug: str, *, use_cache: bool = True) -> dict:
    """Contexto de una ciudad (en caché 60 s para no recalcular entre preguntas seguidas)."""
    if use_cache:
        with _lock:
            hit = _cache.get(slug)
            if hit and hit[0] > time.monotonic():
                return hit[1]
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    def guarded(name: str, fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 — una sección caída no debe tumbar todo el analista
            log.exception("Contexto del analista: falla la sección %s", name)
            return None

    normality = guarded("normalidad", lambda: _normality(analyze_normality(session, slug)))
    regime = guarded("regimen", lambda: _regime(analyze_regimes(session, slug)))

    def _fc():
        try:
            preds = live_forecasts(session, slug)
        except ValueError:
            preds = []
        return _forecast(reliability_report(session, loc, preds))

    forecast = guarded("pronostico", _fc)
    evaluation = guarded("evaluacion", lambda: _evaluation(evaluation_summary(session, loc, 30)))
    models = guarded("modelos", lambda: _models(session, loc.id))
    current = guarded("actual", lambda: _current(session, loc.id))

    ctx = {
        "ciudad": {"nombre": loc.name, "slug": loc.slug, "pais": loc.country,
                   "zona_horaria": loc.timezone},
        "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "disponibilidad": {
            "observacion_actual": current is not None, "normalidad": normality is not None,
            "regimen": regime is not None, "pronostico": forecast is not None,
            "evaluacion_de_pronosticos": evaluation is not None, "modelos": models is not None,
        },
        "observacion_actual": current, "normalidad": normality, "regimen": regime,
        "pronostico": forecast, "evaluacion_pronosticos": evaluation, "modelos": models,
    }
    if use_cache:
        with _lock:
            _cache[slug] = (time.monotonic() + CONTEXT_TTL_SECONDS, ctx)
    return ctx


def clear_cache() -> None:
    with _lock:
        _cache.clear()
