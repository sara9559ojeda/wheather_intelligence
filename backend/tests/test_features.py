"""Pruebas del constructor de features, con foco en **data leakage** (fase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.services.features.builder import (
    add_targets,
    build_feature_frame,
    list_feature_columns,
    target_column,
)


def _series(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": np.cumsum(rng.normal(0, 0.5, n)) + 15,
            "relative_humidity_2m": rng.uniform(30, 90, n),
            "surface_pressure": rng.normal(945, 5, n),
            "wind_speed_10m": rng.uniform(0, 20, n),
            "wind_direction_10m": rng.uniform(0, 360, n),
            "shortwave_radiation": rng.uniform(0, 800, n),
            "cloud_cover": rng.uniform(0, 100, n),
            "precipitation": rng.choice([0, 0, 0, 1.2], n),
            "is_day": rng.integers(0, 2, n),
        }
    )


def test_lag_uses_only_the_past():
    df = build_feature_frame(_series())
    # temperature_2m_lag_1h en la fila i == temperature_2m en la fila i-1
    assert np.allclose(
        df["temperature_2m_lag_1h"].to_numpy()[1:],
        df["temperature_2m"].to_numpy()[:-1],
    )
    assert np.allclose(
        df["temperature_2m_lag_24h"].to_numpy()[24:],
        df["temperature_2m"].to_numpy()[:-24],
    )


def test_rolling_window_excludes_current_instant():
    df = build_feature_frame(_series(), drop_incomplete=False)
    # roll_mean_3h en t = media de {t-3, t-2, t-1}, sin t
    expected = df["temperature_2m"].shift(1).rolling(3, min_periods=3).mean()
    assert np.allclose(
        df["temperature_2m_roll_mean_3h"].to_numpy(),
        expected.to_numpy(),
        equal_nan=True,
    )
    # y desde luego NO coincide con incluir el instante actual
    with_current = df["temperature_2m"].rolling(3, min_periods=3).mean()
    assert not np.allclose(
        df["temperature_2m_roll_mean_3h"].to_numpy()[5:],
        with_current.to_numpy()[5:],
    )


def test_cyclical_encoding_is_on_unit_circle():
    df = build_feature_frame(_series())
    assert np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1.0)
    assert np.allclose(df["wind_dir_sin"] ** 2 + df["wind_dir_cos"] ** 2, 1.0)


def test_targets_point_to_the_future():
    df = add_targets(build_feature_frame(_series()))
    for h in (1, 3, 24):
        col = target_column(h)
        assert np.allclose(
            df[col].to_numpy()[:-h], df["temperature_2m"].to_numpy()[h:]
        )
        assert df[col].isna().tail(h).all()  # las últimas h filas no tienen target


def test_future_values_do_not_leak_into_present_features():
    """Corromper una observación futura NO debe alterar las features de filas previas."""
    base = _series(seed=1)
    feats_a = build_feature_frame(base).set_index("timestamp")

    corrupted = base.copy()
    corrupted.loc[200:, "temperature_2m"] += 999  # destroza el futuro (fila 200 en adelante)
    feats_b = build_feature_frame(corrupted).set_index("timestamp")

    cols = list_feature_columns(feats_a.reset_index())
    # comparar solo filas anteriores a la fila 190 (bien lejos de la corrupción)
    cutoff = feats_a.index[150]
    a = feats_a.loc[feats_a.index < cutoff, cols]
    b = feats_b.loc[feats_b.index < cutoff, cols]
    pd.testing.assert_frame_equal(a, b)


def test_gap_in_series_produces_nan_lags_not_wrong_values():
    df = _series(n=100)
    df = pd.concat([df.iloc[:40], df.iloc[60:]], ignore_index=True)  # hueco de 20 h
    feats = build_feature_frame(df, drop_incomplete=False).set_index("timestamp")
    gap_start = df["timestamp"].iloc[40]
    # justo después del hueco, el lag de 1 h debe ser NaN (no la hora previa al hueco)
    assert pd.isna(feats.loc[gap_start, "temperature_2m_lag_1h"])
