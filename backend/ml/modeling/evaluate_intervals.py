"""Evalúa la cobertura de los intervalos predictivos en el tramo de PRUEBA (offline).

Los intervalos de cada campeón son ``predicción + [p05, p95]`` de los residuales
de validación (nivel nominal 90 %). Aquí se comprueba, sobre el tramo de prueba
(posterior en el tiempo, que el modelo no usó ni para entrenar ni para elegir):

- cobertura empírica y ancho medio del intervalo vigente;
- cobertura por año del tramo de prueba (¿se mantiene o se deteriora?);
- alternativa conformal simétrica calibrada con validación (radio ``q̂``), para
  comparar su cobertura y ancho.

El resultado se guarda en ``model_runs.metrics['interval_eval']`` del campeón
activo. No toca los artefactos entrenados. Es una **evaluación histórica**, no
una garantía de cobertura futura.

Uso:
    python -m backend.ml.modeling.evaluate_intervals --slug bogota
    python -m backend.ml.modeling.evaluate_intervals --all
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.app.db.models import ModelRun
from backend.app.db.queries import get_location
from backend.app.db.session import session_scope
from backend.app.services.features.builder import HORIZONS_HOURS, target_column
from backend.ml.config_loader import (
    ARTIFACTS_DIR,
    DEFAULT_SLUG,
    active_slug,
    load_catalog,
    model_artifact_name,
    processed_dir,
)

log = logging.getLogger("modeling.intervals")
NOMINAL = 0.90


def _evaluate_horizon(slug: str, horizon: int) -> dict | None:
    path = ARTIFACTS_DIR / model_artifact_name(horizon, slug)
    folder = processed_dir(slug)
    if not path.exists() or not (folder / "test.parquet").exists():
        return None
    art = joblib.load(path)
    tcol = target_column(horizon)

    def predict(split: str) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        df = pd.read_parquet(folder / f"{split}.parquet").dropna(subset=[tcol])
        pred = art["pipeline"].predict(df[art["feature_columns"]])
        return pred, df[tcol].to_numpy(dtype="float64"), pd.to_datetime(df["timestamp"], utc=True)

    p_te, y_te, t_te = predict("test")
    p_va, y_va, _ = predict("valid")
    lo, hi = art["residual_p05"], art["residual_p95"]
    inside = (y_te >= p_te + lo) & (y_te <= p_te + hi)

    # conformal simétrico con la corrección de muestra finita, calibrado en validación
    abs_res = np.sort(np.abs(y_va - p_va))
    rank = min(len(abs_res), math.ceil((len(abs_res) + 1) * NOMINAL))
    q_hat = float(abs_res[rank - 1])
    inside_sym = np.abs(y_te - p_te) <= q_hat

    by_year = pd.DataFrame({"year": t_te.dt.year.to_numpy(), "inside": inside}) \
        .groupby("year")["inside"].agg(["mean", "size"])
    return {
        "nominal": NOMINAL, "method": "validation_residual_quantiles",
        "n_test": int(len(y_te)), "coverage_test": round(float(inside.mean()), 4),
        "mean_width_test": round(float(hi - lo), 3),
        "coverage_by_year": {int(y): {"coverage": round(float(r["mean"]), 4), "n": int(r["size"])}
                             for y, r in by_year.iterrows()},
        "conformal_symmetric": {
            "radius": round(q_hat, 3), "coverage_test": round(float(inside_sym.mean()), 4),
            "mean_width_test": round(2 * q_hat, 3), "n_calibration": int(len(abs_res)),
        },
        "model_trained_at": art.get("trained_at"),
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


def evaluate_location(slug: str) -> int:
    done = 0
    with session_scope() as session:
        loc = get_location(session, slug)
        if loc is None:
            log.warning("Ubicación %s no encontrada", slug)
            return 0
        for h in HORIZONS_HOURS:
            result = _evaluate_horizon(slug, h)
            run = session.scalar(
                select(ModelRun).where(
                    ModelRun.location_id == loc.id, ModelRun.horizon_hours == h,
                    ModelRun.is_active.is_(True), ModelRun.model_type.notlike("baseline%"),
                ).order_by(ModelRun.id.desc()).limit(1)
            )
            if result is None or run is None:
                log.warning("[%s] H=%d: sin artefacto/datos de prueba/run activo", slug, h)
                continue
            run.metrics = {**(run.metrics or {}), "interval_eval": result}   # reasigna → JSONB
            log.info("[%s] H=%2dh cobertura test=%.1f %% (nominal %.0f %%) ancho=%.2f °C",
                     slug, h, 100 * result["coverage_test"], 100 * NOMINAL,
                     result["mean_width_test"])
            done += 1
    return done


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    catalog = list(load_catalog())
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()    # sin argumentos: la ciudad de WI_PIPELINE_LOCATION
    g.add_argument("--slug", choices=[DEFAULT_SLUG, *catalog])
    g.add_argument("--all", action="store_true", help="Madrid y todas las del catálogo")
    args = p.parse_args()
    for slug in ([DEFAULT_SLUG, *catalog] if args.all else [args.slug or active_slug()]):
        evaluate_location(slug)


if __name__ == "__main__":
    main()
