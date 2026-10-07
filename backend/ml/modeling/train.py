"""Entrenamiento y comparación de modelos por horizonte (fase 5).

Procedimiento (fase 1, §14.4), **sin mirar test hasta el final**:

    1. entrenar cada modelo en TRAIN
    2. medir todos los modelos + baselines en VALIDACIÓN
    3. campeón = modelo de ML con menor RMSE de validación
    4. comprobar que el campeón supera a la persistencia (para H ≥ 3 h)
    5. UNA evaluación en TEST con el campeón
    6. intervalo de incertidumbre a partir de los residuales de validación
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from backend.app.services.features.builder import target_column
from backend.ml.modeling.baselines import PersistenceBaseline, SeasonalBaseline
from backend.ml.modeling.metrics import all_metrics, skill_vs_reference
from backend.ml.modeling.models import MODEL_TYPES, RANDOM_STATE, build_model, hyperparams

log = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    horizon: int
    validation: dict[str, dict]          # nombre modelo -> métricas de validación
    test_champion: dict                  # métricas de test del campeón
    champion_type: str
    champion: Pipeline
    cv_rmse: list[float]                 # RMSE por fold (walk-forward en train)
    residual_p05: float
    residual_p95: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    beats_persistence: bool = False


def _xy(df: pd.DataFrame, feature_cols: list[str], tcol: str):
    d = df.dropna(subset=[tcol])
    return d, d[feature_cols], d[tcol].to_numpy(dtype="float64")


def _walk_forward_rmse(
    train_df: pd.DataFrame, feature_cols: list[str], tcol: str, model_type: str, n_splits: int = 3
) -> list[float]:
    d, X, y = _xy(train_df, feature_cols, tcol)
    X = X.to_numpy()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for tr_idx, va_idx in tscv.split(X):
        pipe = build_model(model_type).fit(X[tr_idx], y[tr_idx])
        pred = pipe.predict(X[va_idx])
        out.append(round(float(np.sqrt(np.mean((y[va_idx] - pred) ** 2))), 4))
    return out


def _feature_importance(
    pipe: Pipeline, X_valid: pd.DataFrame, y_valid: np.ndarray, feature_cols: list[str]
) -> dict[str, float]:
    """Importancia por permutación en validación (funciona con cualquier modelo).

    Se usa una muestra para acotar el coste (permutar cada variable × repeticiones).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(6000, len(X_valid))
    idx = rng.choice(len(X_valid), n, replace=False)
    r = permutation_importance(
        pipe, X_valid.iloc[idx], y_valid[idx],
        n_repeats=5, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error",
    )
    order = np.argsort(r.importances_mean)[::-1][:15]
    return {feature_cols[i]: round(float(r.importances_mean[i]), 4) for i in order}


def train_horizon(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    seasonal: SeasonalBaseline,
) -> HorizonResult:
    tcol = target_column(horizon)
    tr, Xtr, ytr = _xy(train_df, feature_cols, tcol)
    va, Xva, yva = _xy(valid_df, feature_cols, tcol)
    te, Xte, yte = _xy(test_df, feature_cols, tcol)
    log.info("H=%2dh | train=%d valid=%d test=%d", horizon, len(tr), len(va), len(te))

    results: dict[str, dict] = {
        "baseline_persistence": all_metrics(yva, PersistenceBaseline().predict(va)),
        "baseline_seasonal": all_metrics(yva, seasonal.predict(va)),
    }
    fitted: dict[str, Pipeline] = {}
    for mt in MODEL_TYPES:
        pipe = build_model(mt).fit(Xtr, ytr)
        fitted[mt] = pipe
        results[mt] = all_metrics(yva, pipe.predict(Xva))
        log.info("  %-24s valid RMSE=%.3f  MAE=%.3f  R2=%.3f",
                 mt, results[mt]["rmse"], results[mt]["mae"], results[mt]["r2"])

    champion_type = min(MODEL_TYPES, key=lambda m: results[m]["rmse"])
    champion = fitted[champion_type]

    rmse_pers = results["baseline_persistence"]["rmse"]
    beats = results[champion_type]["rmse"] < rmse_pers

    test_metrics = all_metrics(yte, champion.predict(Xte))
    test_metrics["skill_vs_persistence"] = skill_vs_reference(
        test_metrics["rmse"], all_metrics(yte, PersistenceBaseline().predict(te))["rmse"]
    )

    resid = yva - champion.predict(Xva)
    cv = _walk_forward_rmse(train_df, feature_cols, tcol, champion_type)

    return HorizonResult(
        horizon=horizon,
        validation=results,
        test_champion=test_metrics,
        champion_type=champion_type,
        champion=champion,
        cv_rmse=cv,
        residual_p05=round(float(np.percentile(resid, 5)), 3),
        residual_p95=round(float(np.percentile(resid, 95)), 3),
        feature_importance=_feature_importance(champion, Xva, yva, feature_cols),
        beats_persistence=bool(beats),
    )


def model_run_payload(res: HorizonResult, model_type: str, is_champion: bool) -> dict:
    """Fila para la tabla model_runs (una por modelo × horizonte)."""
    v = res.validation[model_type]
    payload = {
        "model_type": model_type,
        "horizon_hours": res.horizon,
        "target": "temperature_2m",
        "mae": v["mae"],
        "rmse": v["rmse"],
        "r2": v["r2"],
        "hyperparams": (
            hyperparams(model_type) if model_type in MODEL_TYPES else {"kind": "baseline"}
        ),
        "metrics": {"validation": v},
        "is_active": is_champion,
    }
    if is_champion:
        payload["metrics"]["test"] = res.test_champion
        payload["metrics"]["cv_rmse_folds"] = res.cv_rmse
        payload["metrics"]["residual_interval_p05_p95"] = [res.residual_p05, res.residual_p95]
        payload["metrics"]["feature_importance_top"] = res.feature_importance
        payload["metrics"]["beats_persistence"] = res.beats_persistence
    return payload
