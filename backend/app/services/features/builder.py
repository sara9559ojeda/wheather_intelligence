"""Construcción del conjunto de *features* a partir de observaciones horarias.

Entrada esperada: DataFrame con columna ``timestamp`` (UTC, tz-aware) y las
columnas crudas de ``weather_observations``. La serie se **reindexa a horaria
completa** antes de calcular retardos: si hubiera un hueco, los retardos que lo
cruzan quedan como NaN en vez de mezclar horas equivocadas.

Todas las operaciones (``shift`` positivo, ``rolling`` con ventana hacia atrás)
usan solo el pasado. Ver test ``backend/tests/test_features.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TS = "timestamp"

# Horizontes de predicción del proyecto (fase 1, §14).
HORIZONS_HOURS: tuple[int, ...] = (1, 3, 6, 12, 24)

# --- especificación de las features derivadas -------------------------------
_LAGS: dict[str, tuple[int, ...]] = {
    "temperature_2m": (1, 2, 3, 6, 24),
    "relative_humidity_2m": (1, 3, 24),
    "surface_pressure": (1, 3),
    "wind_speed_10m": (1, 3),
    "shortwave_radiation": (1, 24),
    "cloud_cover": (1,),
}
_ROLLINGS: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] = {
    # columna -> ((ventana_horas, (funciones,)), ...)
    "temperature_2m": ((3, ("mean", "std")), (24, ("mean", "std"))),
    "wind_speed_10m": ((3, ("mean",)),),
    "precipitation": ((3, ("sum",)), (24, ("sum",))),
}
_DELTAS: dict[str, int] = {
    # columna -> horas (valor_t - valor_{t-h}); tendencia
    "temperature_2m": 1,
    "relative_humidity_2m": 1,
    "surface_pressure": 3,
}


def _reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TS).drop_duplicates(TS)
    full = pd.date_range(df[TS].min(), df[TS].max(), freq="h", tz="UTC")
    return df.set_index(TS).reindex(full).rename_axis(TS).reset_index()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TS].dt
    df["year"] = ts.year
    df["month"] = ts.month
    df["day"] = ts.day
    df["hour"] = ts.hour
    df["dayofweek"] = ts.dayofweek
    df["dayofyear"] = ts.dayofyear
    df["is_weekend"] = (ts.dayofweek >= 5).astype("int8")
    # codificación cíclica: la hora 23 y la 0 quedan contiguas
    df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    return df


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_direction_10m" in df:
        rad = np.deg2rad(df["wind_direction_10m"].astype("float64"))
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, lags in _LAGS.items():
        if col not in df:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, specs in _ROLLINGS.items():
        if col not in df:
            continue
        for window, funcs in specs:
            # shift(1): la ventana termina en t-1 -> nunca incluye el instante actual
            base = df[col].shift(1).rolling(window=window, min_periods=window)
            for fn in funcs:
                df[f"{col}_roll_{fn}_{window}h"] = getattr(base, fn)()
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, hours in _DELTAS.items():
        if col in df:
            df[f"{col}_delta_{hours}h"] = df[col] - df[col].shift(hours)
    return df


def add_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    if "precipitation" in df:
        df["is_raining"] = (df["precipitation"].fillna(0) > 0).astype("int8")
    if "is_day" in df:
        df["is_day"] = df["is_day"].astype("float64")  # 0.0/1.0 para el modelo
    return df


# Columnas crudas de la observación (no son features del modelo por sí solas;
# algunas se usan vía sus lags/deltas). ``timestamp`` y ``weather_code`` aparte.
_RAW_OBSERVATION_COLUMNS = {
    "timestamp", "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "surface_pressure", "pressure_msl", "precipitation",
    "rain", "snowfall", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_speed_100m", "shortwave_radiation", "direct_radiation",
    "diffuse_radiation", "et0_fao_evapotranspiration", "weather_code",
}

# Columnas de calendario que sí son features aunque no lleven sufijo _lag_/_roll_.
_CALENDAR_FEATURES = {
    "month", "hour", "dayofweek", "dayofyear", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
}


def build_feature_frame(df: pd.DataFrame, *, drop_incomplete: bool = True) -> pd.DataFrame:
    """Devuelve ``df`` + todas las features derivadas.

    Con ``drop_incomplete`` (por defecto) elimina las primeras filas que no
    tienen suficiente historia para los retardos/medias móviles.
    """
    df = _reindex_hourly(df.copy())
    df = add_calendar_features(df)
    df = add_wind_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_delta_features(df)
    df = add_binary_features(df)

    if drop_incomplete:
        needed = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_delta_" in c]
        df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def list_feature_columns(df: pd.DataFrame) -> list[str]:
    """Nombres de las columnas que son *features* del modelo en ``df``."""
    suffixes = ("_lag_", "_roll_", "_delta_")
    out = []
    for c in df.columns:
        if c.startswith("target_temp_h"):
            continue
        if c in _CALENDAR_FEATURES or c in {"is_day", "is_raining", "wind_dir_sin", "wind_dir_cos"}:
            out.append(c)
        elif any(s in c for s in suffixes):
            out.append(c)
    return out


# --- targets (solo para entrenar) ------------------------------------------
def target_column(horizon_hours: int) -> str:
    return f"target_temp_h{horizon_hours}"


def add_targets(
    df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS_HOURS
) -> pd.DataFrame:
    """Añade ``target_temp_h{H}`` = temperatura en ``t + H`` (futuro).

    Las filas del final no tienen target para todos los horizontes: quedan NaN
    y se descartan por-horizonte en el entrenamiento (fase 5), no aquí.
    """
    df = df.sort_values(TS).reset_index(drop=True)
    for h in horizons:
        df[target_column(h)] = df["temperature_2m"].shift(-h)
    return df
