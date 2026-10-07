"""Servicio de predicción de temperatura (inferencia).

Carga los artefactos ``model_temp_h{H}.joblib`` producidos en la fase 5 y
predice para todos los horizontes a partir de las observaciones recientes.
Construye las features con el **mismo** código que el entrenamiento
(``app.services.features``) → sin *train-serving skew*.
"""

from backend.app.services.prediction.predictor import TemperaturePredictor

__all__ = ["TemperaturePredictor"]
