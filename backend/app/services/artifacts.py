"""Carga perezosa y cacheada de los artefactos de ML (fases 4 y 5).

Los artefactos son ficheros ``.joblib`` en ``backend/ml/artifacts/``. Se cargan
una sola vez por proceso. Si faltan, las funciones devuelven ``None`` y los
endpoints correspondientes responden 503 (no rompen toda la API).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib

from backend.app.core.config import settings
from backend.ml.config_loader import ARTIFACTS_DIR

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_predictor():
    from backend.app.services.prediction import TemperaturePredictor

    try:
        return TemperaturePredictor(ARTIFACTS_DIR, settings.wi_location_slug)
    except FileNotFoundError as exc:
        log.warning("Predictor no disponible: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_cluster_artifact() -> dict | None:
    path = ARTIFACTS_DIR / f"kmeans_{settings.wi_location_slug}.joblib"
    if not path.exists():
        log.warning("Artefacto de clustering no encontrado: %s", path)
        return None
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_anomaly_artifact() -> dict | None:
    path = ARTIFACTS_DIR / f"iforest_{settings.wi_location_slug}.joblib"
    if not path.exists():
        log.warning("Artefacto de anomalías no encontrado: %s", path)
        return None
    return joblib.load(path)
