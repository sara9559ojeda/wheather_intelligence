"""Definición de los modelos de regresión a comparar (fase 5).

- **ridge**: regresión lineal regularizada. Techo de lo que explica una relación
  lineal. Necesita escalado → va dentro de un `Pipeline` con `StandardScaler`
  **ajustado solo con train** (el `Pipeline` lo garantiza).
- **random_forest**: bosque aleatorio. Captura no linealidades e interacciones,
  da importancia de variables, robusto. Los árboles no necesitan escalado.
- **hist_gradient_boosting**: boosting por histogramas (la implementación moderna
  y rápida de Gradient Boosting en scikit-learn). Suele ser el mejor en datos
  tabulares. Justifica sustituir a `GradientBoostingRegressor`: mismo método,
  ~50× más rápido en 160 000 filas.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_TYPES: tuple[str, ...] = ("ridge", "random_forest", "hist_gradient_boosting")


def build_model(model_type: str) -> Pipeline:
    if model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ])
    if model_type == "random_forest":
        # Tamaño acotado: con 200 árboles y min_samples_leaf=5 el artefacto llegaba
        # a ~300 MB (Barranquilla H=24h), por encima del límite de 100 MB por
        # archivo de GitHub y del de 500 MB de la función de Vercel. Con esta
        # configuración pesa ~15 MB y pierde ~1,7 % de RMSE en validación.
        return Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_features="sqrt",
                min_samples_leaf=30,
                max_samples=0.5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ])
    if model_type == "hist_gradient_boosting":
        return Pipeline([
            ("model", HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.05,
                max_depth=8,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Modelo desconocido: {model_type}")


def hyperparams(model_type: str) -> dict:
    """Resumen de hiperparámetros (para persistir en model_runs)."""
    est = build_model(model_type).named_steps["model"]
    keep = {
        "ridge": ["alpha"],
        "random_forest": ["n_estimators", "max_features", "min_samples_leaf", "max_samples"],
        "hist_gradient_boosting": ["max_iter", "learning_rate", "max_depth", "l2_regularization"],
    }[model_type]
    return {k: getattr(est, k) for k in keep}
