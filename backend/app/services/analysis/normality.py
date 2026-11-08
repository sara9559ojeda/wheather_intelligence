"""¿Qué tan normal es el tiempo de hoy? — contexto histórico por ciudad.

Para una observación calcula, **sobre observaciones comparables de la misma
ciudad**, el percentil de cada variable, si es un valor extremo univariable, si
el Isolation Forest (modelo multivariable ya entrenado) la marca como anómala y
con qué frecuencia hubo condiciones parecidas.

Rigor estadístico (documentado en ``docs/fase-11-inteligencia-contextual.md``):

- **Comparables** = mismo mes del año y misma franja horaria **en hora local de
  la ciudad** (no se mezclan estaciones ni horas del día). Solo observaciones
  ``historical`` (reanálisis ERA5), nunca la propia serie en tiempo real.
- **Sin fuga de futuro**: la referencia son únicamente observaciones con
  ``observed_at < as_of`` (por defecto, el instante analizado).
- **Respaldo para pocos datos**: si la ventana estricta no alcanza ``MIN_SAMPLE``
  observaciones, se amplía por pasos (``LADDER``) y se indica la menor
  representatividad. Si ni la más amplia alcanza, el estado es ``insufficient``
  y no se emite ninguna conclusión.
- **Percentil** = % de la referencia con valor **menor o igual** al actual.
- **Precipitación**: la mayoría de las horas son secas, así que el percentil
  solo tiene sentido entre las horas con lluvia, y solo con muestra suficiente.
- Un valor estadísticamente extremo **no se presenta como peligroso**: indica
  rareza respecto al histórico, no un fenómeno que ponga en riesgo a nadie.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import Integer, cast, extract, func, select
from sqlalchemy.orm import Session

from backend.app.db.models import WeatherObservation
from backend.app.db.queries import get_location, latest_observation
from backend.app.services.historical.comparison import score_anomaly

log = logging.getLogger(__name__)

MIN_SAMPLE = 300          # observaciones mínimas de referencia para emitir conclusiones
MIN_WET_SAMPLE = 30       # horas con lluvia mínimas para un percentil de intensidad
WET_THRESHOLD_MM = 0.1    # resolución de Open-Meteo: por debajo es "seco"
EXTREME_LOW, EXTREME_HIGH = 2.5, 97.5
CACHE_TTL_SECONDS = 6 * 3600
CACHE_MAX_ENTRIES = 24


@dataclass(frozen=True)
class VarSpec:
    key: str
    label: str
    unit: str
    decimals: int
    in_anomaly_model: bool = False
    upper_only: bool = False   # solo el extremo ALTO es noticia (viento, lluvia)


VARIABLES: tuple[VarSpec, ...] = (
    VarSpec("temperature_2m", "Temperatura", "°C", 1, True),
    VarSpec("relative_humidity_2m", "Humedad relativa", "%", 0, True),
    VarSpec("wind_speed_10m", "Velocidad del viento", "km/h", 1, True, True),
    VarSpec("precipitation", "Precipitación", "mm", 1, False, True),
    VarSpec("surface_pressure", "Presión en superficie", "hPa", 0, True),
)
_BY_KEY = {v.key: v for v in VARIABLES}


@dataclass(frozen=True)
class Step:
    index: int
    month_radius: int       # 0 = solo el mes; 6 = todo el año
    hour_radius: int
    representativeness: str
    description: str


LADDER: tuple[Step, ...] = (
    Step(0, 0, 1, "alta", "mismo mes y misma hora local ±1 h"),
    Step(1, 1, 1, "alta", "mes ±1 y hora local ±1 h"),
    Step(2, 1, 3, "media", "mes ±1 y hora local ±3 h"),
    Step(3, 2, 3, "media", "mes ±2 y hora local ±3 h"),
    Step(4, 6, 3, "baja", "todo el año y hora local ±3 h"),
)

# ---------------------------------------------------------------- caché ligera
_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _cache_get(key: tuple) -> pd.DataFrame | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        _cache.pop(key, None)
    return None


def _cache_put(key: tuple, df: pd.DataFrame) -> None:
    with _cache_lock:
        if len(_cache) >= CACHE_MAX_ENTRIES:
            _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
        _cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, df)


# ---------------------------------------------------------------- referencia
def _window(center: int, radius: int, modulo: int, first: int) -> list[int]:
    """Valores en [first, first+modulo) a distancia ≤ radius de ``center`` (circular)."""
    if 2 * radius + 1 >= modulo:
        return list(range(first, first + modulo))
    return sorted({((center - first + d) % modulo) + first for d in range(-radius, radius + 1)})


def _as_of_key(as_of: datetime) -> str:
    # Con un as_of reciente la referencia es "todo el histórico": una sola entrada de caché.
    if (datetime.now(UTC) - as_of).days < 3:
        return "live"
    return as_of.date().isoformat()


def load_reference(
    session: Session, location_id: int, tz: str, step: Step,
    local_month: int, local_hour: int, as_of: datetime,
) -> pd.DataFrame:
    """Observaciones históricas comparables (solo anteriores a ``as_of``)."""
    key = (location_id, step.index, local_month, local_hour, _as_of_key(as_of))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    local = func.timezone(tz, WeatherObservation.observed_at)
    cols = [getattr(WeatherObservation, v.key) for v in VARIABLES]
    stmt = select(WeatherObservation.observed_at, *cols).where(
        WeatherObservation.location_id == location_id,
        WeatherObservation.source == "historical",
        WeatherObservation.observed_at < as_of,
        cast(extract("month", local), Integer).in_(
            _window(local_month, step.month_radius, 12, 1)),
        cast(extract("hour", local), Integer).in_(
            _window(local_hour, step.hour_radius, 24, 0)),
    )
    rows = session.execute(stmt).all()
    df = pd.DataFrame(rows, columns=["observed_at", *[v.key for v in VARIABLES]])
    if not df.empty:
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        # float64: con float32, 0.1 (mm) deja de ser igual a sí mismo y rompe los empates
        df[[v.key for v in VARIABLES]] = df[[v.key for v in VARIABLES]].astype("float64")
    _cache_put(key, df)
    return df


def select_reference(
    session: Session, location_id: int, tz: str,
    local_month: int, local_hour: int, as_of: datetime,
) -> tuple[Step, pd.DataFrame] | None:
    """Primera ventana de ``LADDER`` con muestra suficiente (None si ninguna)."""
    for step in LADDER:
        df = load_reference(session, location_id, tz, step, local_month, local_hour, as_of)
        if int(df["temperature_2m"].notna().sum()) >= MIN_SAMPLE:
            return step, df
    return None


# ---------------------------------------------------------------- estadística
def percentile_of(sorted_values: np.ndarray, x: float) -> float:
    """% de ``sorted_values`` menores o iguales a ``x``."""
    return float(100.0 * np.searchsorted(sorted_values, x, side="right") / len(sorted_values))


def classify(p: float) -> tuple[str, str]:
    if p < EXTREME_LOW:
        return "exceptional_low", "Excepcionalmente bajo"
    if p < 10.0:
        return "low", "Bajo"
    if p < 25.0:
        return "slightly_low", "Algo por debajo de lo habitual"
    if p <= 75.0:
        return "typical", "Habitual"
    if p <= 90.0:
        return "slightly_high", "Algo por encima de lo habitual"
    if p <= EXTREME_HIGH:
        return "high", "Alto"
    return "exceptional_high", "Excepcionalmente alto"


def _fmt(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}".replace(".", ",")


def _pct_text(p: float) -> str:
    if p >= 99.5:
        return "más de 99,5"
    if p < 0.5:
        return "menos de 0,5"
    return f"{p:.0f}"


def _variable_result(spec: VarSpec, value: float | None, ref: pd.DataFrame, ref_desc: str) -> dict:
    base = {"variable": spec.key, "label": spec.label, "unit": spec.unit}
    if value is None or np.isnan(value):
        return {**base, "status": "not_applicable", "n_sample": 0,
                "explanation": f"{spec.label}: la observación actual no incluye este dato."}

    series = ref[spec.key].dropna().to_numpy(dtype="float64")
    wet_fraction = None
    if spec.key == "precipitation":
        wet_fraction = float(100.0 * (series >= WET_THRESHOLD_MM).mean()) if len(series) else None
        series = series[series >= WET_THRESHOLD_MM] if value >= WET_THRESHOLD_MM else series

    if spec.key == "precipitation" and value < WET_THRESHOLD_MM:
        wf = _fmt(wet_fraction, 0) if wet_fraction is not None else "—"
        return {
            **base, "status": "ok", "value": round(float(value), 2), "n_sample": len(series),
            "classification": "dry", "classification_label": "Sin precipitación",
            "wet_fraction": wet_fraction,
            "explanation": (
                f"Ahora no llueve. En observaciones comparables ({ref_desc}) llovió en "
                f"el {wf} % de las horas."
            ),
        }

    min_n = MIN_WET_SAMPLE if spec.key == "precipitation" else MIN_SAMPLE
    if len(series) < min_n:
        return {
            **base, "status": "insufficient", "value": round(float(value), 2),
            "n_sample": len(series), "wet_fraction": wet_fraction,
            "explanation": (
                f"{spec.label}: hay solo {len(series)} observaciones comparables "
                f"(mínimo {min_n}); no se emite conclusión."
            ),
        }

    series.sort()
    p = percentile_of(series, float(value))
    key, label = classify(p)
    if spec.upper_only and key == "exceptional_low":
        key, label = "low", "Bajo"      # viento en calma o llovizna: lo más común, no un extremo
    q10, q25, q50, q75, q90 = (float(np.percentile(series, q)) for q in (10, 25, 50, 75, 90))
    d = spec.decimals
    what = "intensidad de la lluvia" if spec.key == "precipitation" else spec.label.lower()
    scope = " (solo entre horas con lluvia)" if spec.key == "precipitation" else ""
    explanation = (
        f"La {what} actual ({_fmt(float(value), d)} {spec.unit}) se encuentra en el percentil "
        f"{_pct_text(p)} respecto a {len(series)} observaciones comparables{scope}. Esto "
        f"significa que aproximadamente el {_pct_text(p)} % de esos registros presentaron "
        f"valores iguales o inferiores."
        if spec.key != "temperature_2m" else
        f"La temperatura actual ({_fmt(float(value), d)} {spec.unit}) se encuentra en el "
        f"percentil {_pct_text(p)} respecto a {len(series)} observaciones comparables. Esto "
        f"significa que aproximadamente el {_pct_text(p)} % de esos registros presentaron "
        f"temperaturas iguales o inferiores."
    )
    return {
        **base, "status": "ok", "value": round(float(value), 2),
        "percentile": round(p, 1), "classification": key, "classification_label": label,
        "is_extreme": bool(p > EXTREME_HIGH or (p < EXTREME_LOW and not spec.upper_only)),
        "median": round(q50, 2), "usual_range": (round(q10, 2), round(q90, 2)),
        "typical_range": (round(q25, 2), round(q75, 2)), "n_sample": len(series),
        "wet_fraction": wet_fraction, "explanation": explanation,
    }


def _similar(ref: pd.DataFrame, temp: float | None, rh: float | None) -> dict:
    if temp is None or rh is None:
        return {"available": False,
                "explanation": "Sin temperatura o humedad actual no se puede buscar similares."}
    sub = ref[["observed_at", "temperature_2m", "relative_humidity_2m"]].dropna()
    if len(sub) < MIN_SAMPLE:
        return {"available": False,
                "explanation": "Muestra insuficiente para estimar la frecuencia de situaciones "
                               "parecidas."}
    hit = sub[(sub["temperature_2m"].sub(temp).abs() <= 1.0)
              & (sub["relative_humidity_2m"].sub(rh).abs() <= 8.0)]
    years_total = int(sub["observed_at"].dt.year.nunique())
    years_hit = int(hit["observed_at"].dt.year.nunique())
    share = 100.0 * len(hit) / len(sub)
    return {
        "available": True, "n_similar": int(len(hit)), "share_pct": round(share, 2),
        "years_with_similar": years_hit, "years_total": years_total,
        "tolerance": "temperatura ±1 °C y humedad ±8 puntos",
        "explanation": (
            f"Condiciones parecidas (temperatura ±1 °C y humedad ±8 puntos) aparecieron en el "
            f"{_fmt(share, 1)} % de las observaciones comparables, en {years_hit} de "
            f"{years_total} años distintos."
            if len(hit) else
            "No hay en el histórico comparable ninguna observación con temperatura ±1 °C y "
            "humedad ±8 puntos de las actuales."
        ),
    }


def _anomaly_block(
    slug: str, observed_at: datetime, values: dict, variables: list[dict]
) -> dict:
    scored = score_anomaly(slug, observed_at, values)
    extremes = [v for v in variables if v.get("is_extreme")]
    if scored is None:
        # Sin modelo multivariable seguimos pudiendo hablar de extremos univariables.
        kind = "univariate" if extremes else None
        return {
            "available": False, "kind": kind,
            "explanation": (
                "El detector multivariable no está disponible para esta ciudad o faltan datos. "
                + ("Hay valores extremos por variable (ver arriba)." if extremes else "")
            ).strip(),
        }
    multi = bool(scored["is_anomaly"])
    kind = ("both" if multi and extremes else "multivariate" if multi
            else "univariate" if extremes else "none")
    contributors = []
    if multi:
        ranked = sorted(
            (v for v in variables
             if v["status"] == "ok" and v.get("percentile") is not None
             and _BY_KEY[v["variable"]].in_anomaly_model),
            key=lambda v: abs(v["percentile"] - 50), reverse=True,
        )
        contributors = [
            {"variable": v["variable"], "label": v["label"], "percentile": v["percentile"],
             "direction": "above" if v["percentile"] >= 50 else "below"}
            for v in ranked[:2]
        ]
    texts = {
        "none": "No es una anomalía multivariable ni hay valores extremos aislados: la "
                "combinación de variables es compatible con lo habitual para esta época y hora.",
        "multivariate": "El Isolation Forest la marca como inusual: la COMBINACIÓN de variables "
                        "es rara respecto al histórico, aunque ninguna variable sea extrema por sí "
                        "sola.",
        "univariate": "Alguna variable está en un extremo del histórico, pero la combinación de "
                      "variables no es anómala para el Isolation Forest.",
        "both": "Hay valores extremos por variable y además el Isolation Forest marca la "
                "combinación como inusual.",
    }
    text = texts[kind]
    if contributors:
        names = " y ".join(
            f"{c['label'].lower()} ({'por encima' if c['direction'] == 'above' else 'por debajo'}"
            f" de lo habitual, percentil {c['percentile']:.0f})" for c in contributors
        )
        text += f" Las variables más alejadas de lo habitual son {names}. Es una asociación " \
                "estadística, no una causa."
    text += " Inusual no significa peligroso."
    return {"available": True, "is_multivariate_anomaly": multi, "score": scored["anomaly_score"],
            "kind": kind, "contributors": contributors, "explanation": text}


def _summary(variables: list[dict], anomaly: dict, ref: dict) -> str:
    temp = next((v for v in variables if v["variable"] == "temperature_2m"), None)
    if temp is None or temp["status"] != "ok":
        return "No hay datos suficientes para valorar la temperatura actual."
    s = (f"La temperatura actual ({_fmt(temp['value'], 1)} °C) es "
         f"«{temp['classification_label'].lower()}» para esta época y hora en la ciudad "
         f"(percentil {_pct_text(temp['percentile'])}).")
    others = [v for v in variables if v["variable"] != "temperature_2m" and v.get("is_extreme")]
    if others:
        s += " También destacan: " + ", ".join(
            f"{v['label'].lower()} ({v['classification_label'].lower()})" for v in others) + "."
    if anomaly.get("kind") in ("multivariate", "both"):
        s += " La combinación de variables es inusual según el detector de anomalías."
    if ref["representativeness"] != "alta":
        s += f" Representatividad de la referencia: {ref['representativeness']}."
    return s


# ---------------------------------------------------------------- API del servicio
def _unavailable(loc_name: str, slug: str, summary: str, status: str = "unavailable") -> dict:
    return {"status": status, "location": loc_name, "slug": slug, "summary": summary,
            "variables": [], "notes": []}


def analyze_normality(
    session: Session, slug: str, *, observed_at: datetime | None = None,
    values: dict | None = None, as_of: datetime | None = None,
) -> dict:
    """Análisis de normalidad de una observación (por defecto, la más reciente)."""
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    if observed_at is None or values is None:
        obs = latest_observation(session, loc.id)
        if obs is None:
            return _unavailable(loc.name, slug, "Todavía no hay observaciones de esta ciudad.")
        observed_at = obs.observed_at
        values = {k: (float(getattr(obs, k)) if getattr(obs, k) is not None else None)
                  for k in (*[v.key for v in VARIABLES], "cloud_cover", "shortwave_radiation")}
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    as_of = as_of or observed_at   # SIN FUGA: la referencia es estrictamente anterior

    local = observed_at.astimezone(ZoneInfo(loc.timezone))
    picked = select_reference(session, loc.id, loc.timezone, local.month, local.hour, as_of)
    if picked is None:
        return {
            **_unavailable(
                loc.name, slug,
                "No hay suficiente histórico comparable para esta ciudad, época y hora; "
                "no se emite ninguna conclusión.", "insufficient"),
            "observed_at": observed_at,
            "notes": [f"Se exigen al menos {MIN_SAMPLE} observaciones comparables."],
        }
    step, ref = picked

    ts = ref["observed_at"]
    ref_info = {
        "description": step.description, "n_observations": int(len(ref)),
        "period_start": ts.min().to_pydatetime(), "period_end": ts.max().to_pydatetime(),
        "years": int(ts.dt.year.nunique()), "window_step": step.index,
        "representativeness": step.representativeness, "widened": step.index > 0,
        "min_sample": MIN_SAMPLE, "local_month": local.month, "local_hour": local.hour,
        "timezone": loc.timezone,
    }
    variables = [_variable_result(v, values.get(v.key), ref, step.description)
                 for v in VARIABLES]
    anomaly = _anomaly_block(slug, observed_at, values, variables)
    similar = _similar(ref, values.get("temperature_2m"), values.get("relative_humidity_2m"))

    statuses = {v["status"] for v in variables if v["variable"] != "precipitation"}
    status = "ok" if statuses <= {"ok"} else "partial"
    notes = [
        "Percentil = % de observaciones comparables con valor menor o igual al actual.",
        "La referencia usa solo datos históricos anteriores al instante analizado.",
        "Un valor estadísticamente extremo no implica peligro: indica rareza en esta ciudad.",
    ]
    if step.index > 0:
        notes.insert(0, f"La ventana estricta no tenía muestra suficiente; se amplió a "
                        f"«{step.description}» (representatividad {step.representativeness}).")
    return {
        "status": status, "location": loc.name, "slug": slug, "observed_at": observed_at,
        "reference": ref_info, "variables": variables, "anomaly": anomaly, "similar": similar,
        "summary": _summary(variables, anomaly, ref_info), "notes": notes,
    }
