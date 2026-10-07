"""Métricas de regresión para la predicción de temperatura (fase 5)."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error sistemático medio (pred − real). Positivo = el modelo sobreestima."""
    return float(np.mean(y_pred - y_true))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "r2": round(r2(y_true, y_pred), 4),
        "bias": round(bias(y_true, y_pred), 4),
        "n": int(len(y_true)),
    }


def skill_vs_reference(rmse_model: float, rmse_reference: float) -> float:
    """1 − RMSE_modelo / RMSE_referencia.  >0 → mejor que la referencia."""
    if not rmse_reference:
        return float("nan")
    return round(1 - rmse_model / rmse_reference, 4)
