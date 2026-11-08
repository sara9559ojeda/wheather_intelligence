"""Patrones meteorológicos de la ciudad: perfiles interpretables de los regímenes K-Means.

K-Means asigna cada observación al centroide más cercano en el espacio
estandarizado del entrenamiento (``clustering.build_matrix`` + el
``StandardScaler`` guardado). **No es un clasificador probabilístico**: no hay
"probabilidad de pertenecer al régimen X"; solo "se parece más a X" y cuánto
dista. Por eso el servicio habla de *distancia* y de *frecuencias históricas*.

Todo el perfil se deriva de las observaciones asignadas por el propio modelo:

- etiquetas: se componen a partir de dónde cae la mediana del régimen dentro de
  la distribución de la propia ciudad (cálido = mediana por encima del tercil
  superior del clima local), no se asignan a mano;
- estacionalidad: reparto del régimen por mes **local**;
- permanencia: duración de los episodios (rachas de horas consecutivas);
- transiciones: pares «episodio A → episodio B» sobre la serie ordenada en el
  tiempo (se excluyen huecos de datos y la permanencia en el mismo régimen).

El perfil se calcula una vez por modelo y se guarda en ``cluster_models.profile``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import ClusterAssignment, ClusterModel, Location, WeatherObservation
from backend.app.db.queries import get_location, load_observations_df
from backend.app.services.artifacts import get_cluster_artifact
from backend.ml.mining.clustering import CLUSTER_FEATURES, build_matrix

log = logging.getLogger(__name__)

PROFILE_VERSION = 1
WET_THRESHOLD_MM = 0.1
MIN_TRANSITION_SUPPORT = 30       # cambios de régimen mínimos desde un origen para mostrar %
MIN_EPISODES = 10                 # episodios mínimos para hablar de permanencia
TIMELINE_DAYS = 7

MONTHS_ES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre")

META: dict[str, tuple[str, str, int]] = {      # clave -> (etiqueta, unidad, decimales)
    "temperature_2m": ("Temperatura", "°C", 1),
    "relative_humidity_2m": ("Humedad relativa", "%", 0),
    "surface_pressure": ("Presión", "hPa", 0),
    "wind_speed_10m": ("Viento", "km/h", 0),
    "cloud_cover": ("Nubosidad", "%", 0),
    "shortwave_radiation": ("Radiación solar", "W/m²", 0),
    "precipitation": ("Precipitación", "mm", 1),
}


# ---------------------------------------------------------------- etiquetas
def _rank(sorted_all: np.ndarray, value: float) -> float:
    """Fracción de la ciudad con valor ≤ ``value`` (0–1)."""
    return float(np.searchsorted(sorted_all, value, side="right") / len(sorted_all))


def _descriptors(medians: dict, wet_share: float, z: dict, ranks: dict) -> list[tuple[str, float]]:
    """(descriptor, fuerza) ordenados por fuerza. Cada uno sale de los datos del régimen."""
    out: list[tuple[str, float]] = []
    r = ranks
    if r["relative_humidity_2m"] >= 0.67:
        out.append(("húmedo", abs(z["relative_humidity_2m"])))
    elif r["relative_humidity_2m"] <= 0.33:
        out.append(("seco", abs(z["relative_humidity_2m"])))
    if r["cloud_cover"] >= 0.67:
        out.append(("nublado", abs(z["cloud_cover"])))
    elif r["cloud_cover"] <= 0.33:
        out.append(("despejado", abs(z["cloud_cover"])))
    if r["wind_speed_10m"] >= 0.75:
        out.append(("ventoso", abs(z["wind_speed_10m"])))
    elif r["wind_speed_10m"] <= 0.25 and wet_share < 10:
        out.append(("en calma", abs(z["wind_speed_10m"])))
    if wet_share >= 25:
        out.append(("lluvioso", 3.0 + wet_share / 50))
    elif wet_share >= 10:
        out.append(("con lluvias ocasionales", 1.0 + wet_share / 50))
    if medians["shortwave_radiation"] < 5:
        out.append(("nocturno", 2.5))
    elif r["shortwave_radiation"] >= 0.8:
        out.append(("soleado", abs(z["shortwave_radiation"])))
    return sorted(out, key=lambda t: -t[1])


def _compose(temp_word: str, extras: list[str]) -> str:
    parts = [temp_word, *extras]
    text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " y " + parts[-1]
    return text[0].upper() + text[1:]


def _unique_label(temp_word: str, descs: list[tuple[str, float]], used: set[str]) -> str:
    extras = [d for d, _ in descs]
    for n_extra in (2, 1, 3, 0):
        label = _compose(temp_word, extras[:n_extra])
        if label not in used:
            return label
    base = _compose(temp_word, extras[:2])
    i = 2
    while f"{base} ({i})" in used:
        i += 1
    return f"{base} ({i})"


# ---------------------------------------------------------------- perfil
def _fmt(v: float, d: int = 0) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _load_assigned(session: Session, cm: ClusterModel, tz: str) -> pd.DataFrame:
    cols = [getattr(WeatherObservation, f) for f in CLUSTER_FEATURES]
    stmt = (
        select(WeatherObservation.observed_at, *cols,
               ClusterAssignment.cluster_id, ClusterAssignment.distance_to_centroid)
        .join(ClusterAssignment, ClusterAssignment.observation_id == WeatherObservation.id)
        .where(ClusterAssignment.cluster_model_id == cm.id)
        .order_by(WeatherObservation.observed_at)
    )
    df = pd.DataFrame(session.execute(stmt).all(),
                      columns=["observed_at", *CLUSTER_FEATURES, "cluster", "distance"])
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    num = [*CLUSTER_FEATURES, "distance"]
    df[num] = df[num].astype("float64")
    df["cluster"] = df["cluster"].astype("int64")
    df["local_month"] = df["observed_at"].dt.tz_convert(ZoneInfo(tz)).dt.month
    return df


def _runs(df: pd.DataFrame) -> pd.DataFrame:
    """Episodios: rachas de horas consecutivas con el mismo régimen (sin huecos)."""
    gap = df["observed_at"].diff().ne(pd.Timedelta(hours=1))
    new = df["cluster"].ne(df["cluster"].shift()) | gap
    run_id = new.cumsum()
    g = df.groupby(run_id)
    runs = pd.DataFrame({
        "cluster": g["cluster"].first(), "start": g["observed_at"].first(),
        "end": g["observed_at"].last(), "hours": g.size(),
    }).reset_index(drop=True)
    return runs


def _transitions(runs: pd.DataFrame) -> dict:
    """Pares de episodios contiguos (sin hueco entre uno y el siguiente)."""
    nxt = runs.shift(-1)
    contiguous = (nxt["start"] - runs["end"]) == pd.Timedelta(hours=1)
    pairs = pd.DataFrame({"a": runs["cluster"], "b": nxt["cluster"]})[contiguous].dropna()
    counts: dict[str, dict[str, int]] = {}
    for (a, b), n in pairs.astype(int).value_counts().items():
        if a != b:
            counts.setdefault(str(a), {})[str(b)] = int(n)
    out: dict[str, dict] = {}
    for a, row in counts.items():
        total = sum(row.values())
        out[a] = {"total": total, "to": {
            b: {"count": n, "probability": round(n / total, 4)} for b, n in row.items()}}
    return {"min_support": MIN_TRANSITION_SUPPORT, "from": out, "n_pairs": int(len(pairs))}


def build_profile(session: Session, cm: ClusterModel, loc: Location) -> dict:
    df = _load_assigned(session, cm, loc.timezone)
    if df.empty:
        raise ValueError("El modelo de clustering no tiene asignaciones")
    n = len(df)
    feats = list(CLUSTER_FEATURES)
    sorted_all = {f: np.sort(df[f].dropna().to_numpy()) for f in feats}
    mean_all = {f: float(df[f].mean()) for f in feats}
    std_all = {f: float(df[f].std()) or 1.0 for f in feats}
    wet_all = float((df["precipitation"] >= WET_THRESHOLD_MM).mean())
    month_share_all = df["local_month"].value_counts(normalize=True).sort_index()
    runs = _runs(df)

    legacy = {int(k): v for k, v in (cm.cluster_labels or {}).items()}
    ids = sorted(int(c) for c in df["cluster"].unique())
    used: set[str] = set()
    profiles: list[dict] = []
    for cid in ids:
        sub = df[df["cluster"] == cid]
        medians = {f: float(sub[f].median()) for f in feats}
        iqr = {f: (float(sub[f].quantile(0.25)), float(sub[f].quantile(0.75))) for f in feats}
        wet_share = float((sub["precipitation"] >= WET_THRESHOLD_MM).mean() * 100)
        z = {f: (medians[f] - mean_all[f]) / std_all[f] for f in feats if f != "precipitation"}
        wet_sd = max(float(np.sqrt(wet_all * (1 - wet_all))), 1e-6)
        z["precipitation"] = (wet_share / 100 - wet_all) / wet_sd
        ranks = {f: _rank(sorted_all[f], medians[f]) for f in feats}

        t_rank = ranks["temperature_2m"]
        temp_word = "cálido" if t_rank >= 0.67 else "fresco" if t_rank <= 0.33 else "templado"
        descs = _descriptors(medians, wet_share, z, ranks)
        label = _unique_label(temp_word, descs, used)
        used.add(label)

        dominant = sorted(z.items(), key=lambda kv: -abs(kv[1]))
        dominant = [kv for kv in dominant if abs(kv[1]) >= 0.5][:3] or dominant[:1]

        by_month = sub["local_month"].value_counts().reindex(range(1, 13), fill_value=0)
        seasonality = []
        for m in range(1, 13):
            share = 100 * by_month[m] / len(sub)
            p_given_m = by_month[m] / max(month_share_all.get(m, 0) * n, 1)
            seasonality.append({"month": m, "share_pct": round(float(share), 2),
                                "lift": round(float(p_given_m / (len(sub) / n)), 3)})
        peaks = [s["month"] for s in sorted(seasonality, key=lambda s: -s["lift"])[:2]
                 if s["lift"] >= 1.2]

        r = runs[runs["cluster"] == cid]["hours"]
        persistence = None
        if len(r) >= MIN_EPISODES:
            persistence = {
                "n_episodes": int(len(r)), "median_hours": float(r.median()),
                "mean_hours": round(float(r.mean()), 2),
                "p90_hours": float(r.quantile(0.9)),
                "stay_probability_1h": round(float(1 - len(r) / len(sub)), 4),
            }

        share = 100 * len(sub) / n
        d = META
        typical = (f"{_fmt(medians['temperature_2m'], 1)} °C, humedad "
                   f"{_fmt(medians['relative_humidity_2m'])} %, viento "
                   f"{_fmt(medians['wind_speed_10m'])} km/h y nubosidad "
                   f"{_fmt(medians['cloud_cover'])} %")
        rain = (f"llueve en el {_fmt(wet_share)} % de sus horas" if wet_share >= 1
                else "casi nunca llueve")
        when = (" Es más frecuente en " + " y ".join(MONTHS_ES[m - 1] for m in peaks) + "."
                if peaks else " Se reparte durante todo el año.")
        stay = (f" Cuando aparece suele durar unas {_fmt(persistence['median_hours'])} h seguidas."
                if persistence else "")
        description = (f"Tiempo {label.lower()} para esta ciudad. Valores típicos: {typical}; "
                       f"{rain}. Ocupa el {_fmt(share, 1)} % de las horas.{when}{stay}")

        profiles.append({
            "regime_id": cid, "label": label, "description": description,
            "legacy_label": legacy.get(cid),
            "n_observations": int(len(sub)), "share_pct": round(share, 2),
            "medians": {f: round(v, 2) for f, v in medians.items()},
            "iqr": {f: (round(lo, 2), round(hi, 2)) for f, (lo, hi) in iqr.items()},
            "z_vs_city": {f: round(float(v), 2) for f, v in z.items()},
            "dominant": [{"variable": f, "label": d[f][0], "unit": d[f][1],
                          "direction": "above" if v > 0 else "below", "z": round(float(v), 2),
                          "median": round(medians[f], 2)} for f, v in dominant],
            "wet_share_pct": round(wet_share, 2), "seasonality": seasonality,
            "peak_months": peaks, "persistence": persistence,
            "distance_p50": round(float(sub["distance"].quantile(0.5)), 3),
            "distance_p90": round(float(sub["distance"].quantile(0.9)), 3),
            "distance_p99": round(float(sub["distance"].quantile(0.99)), 3),
        })

    return {
        "version": PROFILE_VERSION, "computed_at": datetime.now(UTC).isoformat(),
        "n_observations": n,
        "period": [df["observed_at"].min().isoformat(), df["observed_at"].max().isoformat()],
        "feature_space": ("StandardScaler sobre " + ", ".join(feats)
                          + " (precipitación comprimida con log1p), igual que en el entrenamiento"),
        "labels_rule": ("Etiquetas compuestas con la posición de la mediana de cada régimen "
                        "dentro del clima de la propia ciudad (terciles) y su % de horas "
                        "con lluvia."),
        "regimes": profiles, "transitions": _transitions(runs),
    }


def ensure_profile(session: Session, cm: ClusterModel, loc: Location) -> dict:
    """Perfil persistido; si falta, se calcula y se guarda (el llamador hace commit)."""
    if cm.profile and cm.profile.get("version") == PROFILE_VERSION:
        return cm.profile
    profile = build_profile(session, cm, loc)
    cm.profile = profile
    # Una sola fuente de verdad para las etiquetas en toda la app.
    cm.cluster_labels = {str(p["regime_id"]): p["label"] for p in profile["regimes"]}
    session.flush()
    log.info("Perfil de regímenes calculado para %s (k=%d)", loc.slug, cm.k)
    return profile


# ---------------------------------------------------------------- asignación
def assign_frame(art: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Régimen y distancias de cada fila válida, con el MISMO preprocesamiento del entrenamiento."""
    x = build_matrix(df)
    if x.empty:
        return pd.DataFrame(columns=["regime", "distance", "second", "second_distance"])
    dist_all = art["kmeans"].transform(art["scaler"].transform(x))
    order = np.argsort(dist_all, axis=1)
    rows = np.arange(len(x))
    return pd.DataFrame({
        "regime": order[:, 0], "distance": dist_all[rows, order[:, 0]],
        "second": order[:, 1] if dist_all.shape[1] > 1 else order[:, 0],
        "second_distance": dist_all[rows, order[:, 1]] if dist_all.shape[1] > 1 else np.nan,
    }, index=x.index)


def _band(dist: float, prof: dict) -> str:
    if prof["distance_p90"] is not None and dist <= prof["distance_p90"]:
        return "típica"
    if prof["distance_p99"] is not None and dist <= prof["distance_p99"]:
        return "algo alejada"
    return "atípica"


def _episodes(points: pd.DataFrame, labels: dict[int, str]) -> list[dict]:
    if points.empty:
        return []
    gap = points["t"].diff().ne(pd.Timedelta(hours=1))
    run_id = (points["regime"].ne(points["regime"].shift()) | gap).cumsum()
    out = []
    for _, g in points.groupby(run_id):
        rid = int(g["regime"].iloc[0])
        out.append({"regime_id": rid, "label": labels[rid], "start": g["t"].iloc[0].to_pydatetime(),
                    "end": g["t"].iloc[-1].to_pydatetime(), "hours": int(len(g))})
    return out


def analyze_regimes(session: Session, slug: str) -> dict:
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")
    base = {"location": loc.name, "slug": slug}
    cm = session.scalar(
        select(ClusterModel)
        .where(ClusterModel.is_active, ClusterModel.location_id == loc.id)
        .order_by(ClusterModel.id.desc()).limit(1)
    )
    if cm is None:
        return {**base, "status": "unavailable", "notes": [
            "Esta ciudad todavía no tiene un modelo de regímenes (K-Means) entrenado."]}
    try:
        profile = ensure_profile(session, cm, loc)
    except ValueError as exc:
        return {**base, "status": "unavailable", "notes": [str(exc)]}

    regimes = profile["regimes"]
    by_id = {p["regime_id"]: p for p in regimes}
    labels = {rid: p["label"] for rid, p in by_id.items()}
    notes = [
        "K-Means asigna cada observación al régimen cuyo centroide está más cerca; no calcula "
        "probabilidades de pertenencia.",
        "Las frecuencias de transición son históricas (entre episodios consecutivos de "
        "regímenes distintos), no predicciones.",
    ]
    out: dict = {
        **base, "status": "ok",
        "model": {"k": cm.k, "silhouette": float(cm.silhouette) if cm.silhouette else None,
                  "trained_at": cm.trained_at.isoformat(),
                  "feature_space": profile["feature_space"],
                  "n_observations": profile["n_observations"], "period": profile["period"],
                  "labels_rule": profile["labels_rule"]},
        "regimes": regimes, "notes": notes, "timeline": [], "episodes": [],
        "recent_share_pct": {}, "transitions_from_current": [],
    }

    art = get_cluster_artifact(slug)
    if art is None:
        out["status"] = "partial"
        notes.append("Falta el artefacto del modelo: no se puede situar la observación actual.")
        return out

    since = datetime.now(UTC) - timedelta(days=TIMELINE_DAYS)
    recent = load_observations_df(session, slug, since=since)
    if recent.empty:
        out["status"] = "partial"
        notes.append("No hay observaciones recientes para situar el régimen actual.")
        return out
    recent = recent.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    assigned = assign_frame(art, recent)
    if assigned.empty:
        out["status"] = "partial"
        notes.append("Las observaciones recientes están incompletas: no se puede asignar régimen.")
        return out

    pts = pd.DataFrame({"t": recent.loc[assigned.index, "timestamp"].to_numpy(),
                        "regime": assigned["regime"].to_numpy()})
    pts["t"] = pd.to_datetime(pts["t"], utc=True)
    out["timeline"] = [{"t": t.to_pydatetime(), "regime_id": int(r)}
                       for t, r in zip(pts["t"], pts["regime"], strict=True)]
    out["episodes"] = _episodes(pts, labels)
    shares = pts["regime"].value_counts(normalize=True) * 100
    out["recent_share_pct"] = {int(k): round(float(v), 1) for k, v in shares.items()}

    last = assigned.iloc[-1]
    cid = int(last["regime"])
    prof = by_id[cid]
    dist = float(last["distance"])
    ratio = (dist / float(last["second_distance"])
             if last["second_distance"] and not np.isnan(last["second_distance"]) else None)
    second = int(last["second"]) if int(last["second"]) != cid else None
    band = _band(dist, prof)
    t_obs = pts["t"].iloc[-1].to_pydatetime()
    expl = (f"La última observación se parece más al régimen «{prof['label']}». Su distancia "
            f"al centroide es {_fmt(dist, 2)} en el espacio estandarizado (lo habitual dentro de "
            f"este régimen llega hasta {_fmt(prof['distance_p90'], 2)}), así que es una "
            f"situación {band} del régimen.")
    if ratio is not None and second is not None and ratio > 0.85:
        expl += (f" Está casi a la misma distancia del régimen «{labels[second]}»: la frontera "
                 "entre ambos es difusa.")
    out["current"] = {
        "regime_id": cid, "label": prof["label"], "distance_to_centroid": round(dist, 3),
        "distance_band": band, "second_regime_id": second,
        "second_label": labels.get(second) if second is not None else None,
        "distance_ratio_to_second": round(ratio, 3) if ratio is not None else None,
        "observed_at": t_obs, "explanation": expl,
    }

    trans = profile["transitions"]["from"].get(str(cid))
    if trans and trans["total"] >= MIN_TRANSITION_SUPPORT:
        out["transitions_supported"] = True
        out["transitions_total"] = trans["total"]
        out["transitions_from_current"] = sorted(
            ({"to_regime_id": int(b), "label": labels[int(b)], "count": v["count"],
              "probability": v["probability"]} for b, v in trans["to"].items()),
            key=lambda t: -t["probability"],
        )
    else:
        out["transitions_supported"] = False
        out["transitions_total"] = trans["total"] if trans else 0
        notes.append(f"Hay menos de {MIN_TRANSITION_SUPPORT} cambios históricos desde este "
                     "régimen: no se muestran frecuencias de transición.")
    return out
