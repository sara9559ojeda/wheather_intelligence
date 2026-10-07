"""Pruebas de la minería de datos: clustering y anomalías (fase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ml.mining import anomalies as an
from backend.ml.mining import clustering as clu


def _weather(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2010-01-01", periods=n, freq="h", tz="UTC")
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy()
    season = 10 * np.sin(2 * np.pi * doy / 365.25)
    daily = 5 * np.sin(2 * np.pi * (hour - 3) / 24)
    temp = 14 + season + daily + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "timestamp": ts,
        "temperature_2m": temp,
        "relative_humidity_2m": np.clip(70 - 0.8 * (temp - 14) + rng.normal(0, 8, n), 5, 100),
        "surface_pressure": rng.normal(945, 5, n),
        "wind_speed_10m": np.abs(rng.normal(8, 4, n)),
        "cloud_cover": rng.uniform(0, 100, n),
        "shortwave_radiation": np.clip(daily * 60 + rng.normal(0, 30, n), 0, None),
        "precipitation": rng.choice([0.0, 0.0, 0.0, 1.5], n),
    })


# --- clustering ------------------------------------------------------------
def test_build_matrix_compresses_precipitation():
    df = _weather()
    X = clu.build_matrix(df)
    assert list(X.columns) == list(clu.CLUSTER_FEATURES)
    # log1p: el máximo comprimido es menor que el crudo
    assert X["precipitation"].max() < df["precipitation"].max()


def test_kmeans_assign_returns_valid_clusters_and_distances():
    df = _weather()
    model = clu.fit(df, k=3)
    labels, dist = model.assign(df)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert (dist >= 0).all()
    assert len(labels) == len(df)


def test_label_helper_reacts_to_temperature():
    hot = pd.Series({k: 0.0 for k in clu.CLUSTER_FEATURES})
    hot["temperature_2m"] = 1.5
    cold = hot.copy()
    cold["temperature_2m"] = -1.5
    assert "cálido" in clu._label_from_deviation(hot)
    assert "frío" in clu._label_from_deviation(cold)


# --- anomalías ------------------------------------------------------------
def test_climatology_has_one_row_per_month_hour():
    clim = an.fit_climatology(_weather(n=8760))
    assert clim.shape[0] == 12 * 24
    assert list(clim.columns) == list(an.ANOMALY_VARIABLES)


def test_residual_of_average_value_is_near_zero():
    df = _weather(n=8760)
    clim = an.fit_climatology(df)
    res = an.compute_residuals(df, clim)
    # la media de los residuales debe ser ~0 (se ha quitado la estacionalidad)
    assert abs(res["temperature_2m_resid"].mean()) < 0.1


def test_deseasonalization_removes_seasonal_signal():
    df = _weather(n=8760)
    clim = an.fit_climatology(df)
    res = an.compute_residuals(df, clim).assign(month=df["timestamp"].dt.month)
    by_month = df.assign(m=df["timestamp"].dt.month).groupby("m")["temperature_2m"].mean()
    seasonal_var_raw = by_month.var()
    seasonal_var_resid = res.groupby("month")["temperature_2m_resid"].mean().var()
    assert seasonal_var_resid < 0.05 * seasonal_var_raw


def test_isolation_forest_flags_injected_outliers():
    df = _weather(n=15000)
    df.loc[500, "temperature_2m"] += 30       # calor imposible para la hora
    df.loc[900, "surface_pressure"] -= 45     # presión desplomada
    model = an.fit(df, contamination=0.02)
    scored = model.score(df)

    # las filas inyectadas están entre las más anómalas (peor 2 % de scores)
    threshold = scored["anomaly_score"].quantile(0.02)
    assert scored.loc[500, "anomaly_score"] <= threshold
    assert scored.loc[900, "anomaly_score"] <= threshold
    assert scored["is_anomaly"].mean() < 0.05
