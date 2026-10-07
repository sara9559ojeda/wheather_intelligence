"""Construcción de *features* (variables derivadas).

Este paquete lo usan **tanto** la analítica offline (``backend.ml``, para
entrenar) **como** el servicio de predicción en tiempo real (fase 6). Por eso
vive en ``app.services`` y no en ``ml``: las features de entrenamiento y las de
inferencia deben construirse con **exactamente el mismo código** (evita el
*train-serving skew*).

Regla de oro anti-*data-leakage*: toda variable derivada en el instante ``t``
usa **solo** información de ``t`` o anterior. Los *targets* (futuro) se crean
aparte, con ``add_targets``, y solo para entrenar.
"""

from backend.app.services.features.builder import (
    HORIZONS_HOURS,
    add_targets,
    build_feature_frame,
    list_feature_columns,
    target_column,
)

__all__ = [
    "HORIZONS_HOURS",
    "add_targets",
    "build_feature_frame",
    "list_feature_columns",
    "target_column",
]
