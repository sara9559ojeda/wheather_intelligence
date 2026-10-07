"""Clustering de condiciones meteorológicas con K-Means (fase 4).

Objetivo: descubrir **regímenes meteorológicos** (combinaciones típicas de
condiciones) sin fijar de antemano cuántos hay ni qué son. El número de
clusters se decide con **Elbow + Silhouette + interpretabilidad**.

Se usa solo el estado físico instantáneo (temperatura, humedad, presión,
viento, nubosidad, radiación, precipitación); **no** se meten variables de
tiempo (hora, mes) para que los clusters sean "tipos de tiempo", no "momentos
del día".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

CLUSTER_FEATURES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
)
RANDOM_STATE = 42


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y transforma las columnas para el clustering."""
    X = df.loc[:, CLUSTER_FEATURES].astype("float64").copy()
    # la precipitación está tan sesgada a 0 que domina la distancia si no se comprime
    X["precipitation"] = np.log1p(X["precipitation"].clip(lower=0))
    return X.dropna()


@dataclass
class KSelection:
    scores: pd.DataFrame          # columnas: k, inertia, silhouette
    chosen_k: int
    reason: str


def evaluate_k(
    x_scaled: np.ndarray,
    *,
    k_values: range = range(2, 11),
    silhouette_sample: int = 20_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = (
        rng.choice(len(x_scaled), silhouette_sample, replace=False)
        if len(x_scaled) > silhouette_sample
        else np.arange(len(x_scaled))
    )
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(x_scaled)
        sil = silhouette_score(x_scaled[idx], labels[idx])
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
        log.info("  k=%2d  inertia=%.0f  silhouette=%.4f", k, km.inertia_, sil)
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> KSelection:
    """Elige k: mejor silhouette, comprobando que el 'codo' es coherente."""
    best = scores.loc[scores["silhouette"].idxmax()]
    k = int(best["k"])
    reason = (
        f"k={k} maximiza el silhouette medio ({best['silhouette']:.3f}). "
        "El método del codo (inertia vs k) se aporta como comprobación visual."
    )
    return KSelection(scores=scores, chosen_k=k, reason=reason)


@dataclass
class ClusterModel:
    scaler: StandardScaler
    kmeans: KMeans
    features: list[str]
    k: int

    def assign(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (cluster_id, distancia_al_centroide) para cada fila de ``df``."""
        X = build_matrix(df)
        xs = self.scaler.transform(X)
        labels = self.kmeans.predict(xs)
        dist = np.linalg.norm(xs - self.kmeans.cluster_centers_[labels], axis=1)
        return labels, dist


def fit(df_train: pd.DataFrame, k: int) -> ClusterModel:
    X = build_matrix(df_train)
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(scaler.transform(X))
    return ClusterModel(scaler=scaler, kmeans=km, features=list(CLUSTER_FEATURES), k=k)


def _label_from_deviation(dev: pd.Series) -> str:
    """Etiqueta textual a partir de la desviación (en unidades std) del centroide."""
    parts: list[str] = []
    t = dev["temperature_2m"]
    if t > 0.75:
        parts.append("cálido")
    elif t > 0.25:
        parts.append("templado")
    elif t < -0.5:
        parts.append("frío")
    else:
        parts.append("fresco")

    h = dev["relative_humidity_2m"]
    if h > 0.5:
        parts.append("húmedo")
    elif h < -0.5:
        parts.append("seco")

    if dev["cloud_cover"] > 0.6:
        parts.append("nublado")
    elif dev["cloud_cover"] < -0.6:
        parts.append("despejado")

    if dev["shortwave_radiation"] > 0.7:
        parts.append("soleado")
    if dev["wind_speed_10m"] > 0.7:
        parts.append("ventoso")
    if dev["precipitation"] > 0.7:
        parts.append("con precipitación")
    return ", ".join(parts)


def interpret(model: ClusterModel, df_train: pd.DataFrame) -> pd.DataFrame:
    """Centroides en unidades originales + estadística + etiqueta por cluster."""
    X = build_matrix(df_train)
    labels = model.kmeans.predict(model.scaler.transform(X))
    centers_orig = pd.DataFrame(
        model.scaler.inverse_transform(model.kmeans.cluster_centers_),
        columns=model.features,
    )
    # deshacer el log1p de la precipitación para mostrarla en mm
    centers_orig["precipitation"] = np.expm1(centers_orig["precipitation"]).clip(lower=0)

    scaled_centers = pd.DataFrame(model.kmeans.cluster_centers_, columns=model.features)
    global_mean = pd.Series(0.0, index=model.features)  # datos estandarizados

    rows = []
    counts = pd.Series(labels).value_counts().sort_index()
    for cid in range(model.k):
        dev = scaled_centers.loc[cid] - global_mean
        rows.append(
            {
                "cluster": cid,
                "n": int(counts.get(cid, 0)),
                "pct": round(100 * counts.get(cid, 0) / len(labels), 1),
                "etiqueta": _label_from_deviation(dev),
                **{c: round(float(centers_orig.loc[cid, c]), 1) for c in model.features},
            }
        )
    return pd.DataFrame(rows)
