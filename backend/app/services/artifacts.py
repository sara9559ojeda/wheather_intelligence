"""Carga perezosa y cacheada de los artefactos de ML (fases 4 y 5), por ubicación.

Los artefactos son ficheros ``.joblib`` en ``backend/ml/artifacts/``. Se cargan
una sola vez por proceso y ubicación. Si faltan, las funciones devuelven ``None``
y los endpoints correspondientes responden 503 (no rompen toda la API).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib

from backend.ml.config_loader import ARTIFACTS_DIR

log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_predictor(slug: str):
    from backend.app.services.prediction import TemperaturePredictor

    try:
        return TemperaturePredictor(ARTIFACTS_DIR, slug)
    except FileNotFoundError as exc:
        log.warning("Predictor no disponible para %s: %s", slug, exc)
        return None


@lru_cache(maxsize=32)
def get_cluster_artifact(slug: str) -> dict | None:
    path = ARTIFACTS_DIR / f"kmeans_{slug}.joblib"
    if not path.exists():
        log.warning("Artefacto de clustering no encontrado: %s", path)
        return None
    return joblib.load(path)


@lru_cache(maxsize=32)
def get_anomaly_artifact(slug: str) -> dict | None:
    path = ARTIFACTS_DIR / f"iforest_{slug}.joblib"
    if not path.exists():
        log.warning("Artefacto de anomalías no encontrado: %s", path)
        return None
    return joblib.load(path)
