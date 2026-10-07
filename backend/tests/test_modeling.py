"""Pruebas de la capa de modelado (fase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.app.services.features.builder import target_column
from backend.ml.modeling import metrics as M
from backend.ml.modeling.baselines import PersistenceBaseline, SeasonalBaseline
from backend.ml.modeling.models import MODEL_TYPES, build_model
from backend.ml.modeling.train import train_horizon


def test_metrics_match_manual_computation():
    y = np.array([10.0, 12.0, 14.0, 16.0])
    p = np.array([11.0, 11.0, 15.0, 15.0])
    m = M.all_metrics(y, p)
    assert m["mae"] == 1.0
    assert m["rmse"] == 1.0
    assert m["bias"] == 0.0
    assert m["n"] == 4


def test_skill_score_sign():
    assert M.skill_vs_reference(1.0, 2.0) == 0.5      # mitad de error -> skill 0.5
    assert M.skill_vs_reference(3.0, 2.0) == -0.5     # peor que la referencia


def test_persistence_baseline_returns_current_temperature():
    df = pd.DataFrame({"temperature_2m": [5.0, 6.0, 7.0]})
    assert list(PersistenceBaseline().predict(df)) == [5.0, 6.0, 7.0]


def test_seasonal_baseline_learns_month_hour_table():
    ts = pd.date_range("2010-01-01", periods=24 * 60, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "temperature_2m": np.arange(len(ts)) % 24})
    sb = SeasonalBaseline().fit(df)
    pred = sb.predict(df.head(24))
    assert len(pred) == 24
    assert not np.isnan(pred).any()


def test_ridge_has_scaler_trees_do_not():
    assert "scaler" in build_model("ridge").named_steps
    assert "scaler" not in build_model("random_forest").named_steps


def _synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    temp = 15 + 8 * np.sin(2 * np.pi * ts.dayofyear / 365.25) + rng.normal(0, 1, n)
    df = pd.DataFrame({
        "timestamp": ts,
        "temperature_2m": temp,
        "f_lag_1h": np.r_[np.nan, temp[:-1]],
        "f_roll_mean_3h": pd.Series(temp).shift(1).rolling(3).mean().to_numpy(),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
    })
    for h in (1, 3):
        df[target_column(h)] = df["temperature_2m"].shift(-h)
    return df.dropna().reset_index(drop=True)


def test_train_horizon_selects_a_champion_and_evaluates_test():
    df = _synthetic()
    n = len(df)
    tr, va, te = df.iloc[: int(n * 0.6)], df.iloc[int(n * 0.6):int(n * 0.8)], df.iloc[int(n * 0.8):]
    feats = ["f_lag_1h", "f_roll_mean_3h", "hour_sin"]
    seasonal = SeasonalBaseline().fit(tr)

    res = train_horizon(tr, va, te, feats, horizon=1, seasonal=seasonal)

    assert res.champion_type in MODEL_TYPES
    assert isinstance(res.champion, Pipeline)
    assert set(res.validation) == {"baseline_persistence", "baseline_seasonal", *MODEL_TYPES}
    assert res.test_champion["n"] == len(te)
    assert res.residual_p05 < res.residual_p95
    assert len(res.cv_rmse) == 3


def test_champion_beats_persistence_on_learnable_signal():
    """Con una señal aprendible (lag + tendencia), el ML debe batir a la persistencia."""
    df = _synthetic(n=5000)
    n = len(df)
    tr, va, te = df.iloc[: int(n * 0.6)], df.iloc[int(n * 0.6):int(n * 0.8)], df.iloc[int(n * 0.8):]
    feats = ["f_lag_1h", "f_roll_mean_3h", "hour_sin"]
    res = train_horizon(tr, va, te, feats, 3, SeasonalBaseline().fit(tr))
    champ_rmse = res.validation[res.champion_type]["rmse"]
    pers_rmse = res.validation["baseline_persistence"]["rmse"]
    assert champ_rmse < pers_rmse


def test_predictor_produces_a_prediction_per_horizon():
    """Integración: carga los artefactos reales y predice a partir de observaciones."""
    import pytest

    from backend.app.services.features.builder import HORIZONS_HOURS
    from backend.app.services.prediction import TemperaturePredictor
    from backend.ml.config_loader import ARTIFACTS_DIR, load_config

    cfg = load_config()
    if not (ARTIFACTS_DIR / "model_temp_h1.joblib").exists():
        pytest.skip("faltan artefactos; ejecuta backend.ml.modeling.run_training")

    predictor = TemperaturePredictor(ARTIFACTS_DIR, cfg.slug)
    ts = pd.date_range("2024-06-01", periods=48, freq="h", tz="UTC")
    obs = pd.DataFrame({
        "timestamp": ts,
        "temperature_2m": 20 + 6 * np.sin(2 * np.pi * ts.hour / 24),
        "relative_humidity_2m": 55.0, "dew_point_2m": 10.0, "apparent_temperature": 20.0,
        "surface_pressure": 945.0, "pressure_msl": 1015.0, "precipitation": 0.0,
        "rain": 0.0, "snowfall": 0.0, "cloud_cover": 20.0, "cloud_cover_low": 5.0,
        "cloud_cover_mid": 5.0, "cloud_cover_high": 10.0, "wind_speed_10m": 8.0,
        "wind_direction_10m": 230.0, "wind_gusts_10m": 15.0, "wind_speed_100m": 14.0,
        "shortwave_radiation": 300.0, "direct_radiation": 200.0, "diffuse_radiation": 90.0,
        "et0_fao_evapotranspiration": 0.2, "weather_code": 1, "is_day": 1.0,
    })
    preds = predictor.predict_from_observations(obs)
    assert {p.horizon_hours for p in preds} == set(HORIZONS_HOURS)
    for p in preds:
        assert p.lower_bound <= p.predicted_temperature <= p.upper_bound
        assert -20 < p.predicted_temperature < 50
