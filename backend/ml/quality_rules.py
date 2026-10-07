"""Reglas de validación física para las variables meteorológicas.

Se usan en Data Understanding (fase 2) para contar valores fuera de rango y,
más adelante, en el ETL para rechazar o marcar observaciones inválidas.

Los rangos son "físicamente plausibles para Madrid-Barajas", deliberadamente
amplios: el objetivo es detectar errores groseros (sensor, unidad, signo),
NO recortar extremos meteorológicos reales.
"""

from __future__ import annotations

# ---- Variables Open-Meteo (nombres canónicos del proyecto) -------------------
# (mínimo, máximo) inclusivos. None = sin límite por ese lado.
PLAUSIBLE_RANGES: dict[str, tuple[float | None, float | None]] = {
    "temperature_2m": (-25.0, 48.0),          # récord Madrid ~ -10 / +42 °C
    "relative_humidity_2m": (0.0, 100.0),
    "dew_point_2m": (-35.0, 30.0),
    "apparent_temperature": (-35.0, 55.0),
    "surface_pressure": (905.0, 972.0),        # a ~618 m la presión ronda 946 hPa
    "pressure_msl": (975.0, 1050.0),
    "precipitation": (0.0, 60.0),              # mm/h
    "rain": (0.0, 60.0),
    "snowfall": (0.0, 30.0),                   # cm/h
    "cloud_cover": (0.0, 100.0),
    "cloud_cover_low": (0.0, 100.0),
    "cloud_cover_mid": (0.0, 100.0),
    "cloud_cover_high": (0.0, 100.0),
    "wind_speed_10m": (0.0, 130.0),            # km/h
    "wind_direction_10m": (0.0, 360.0),
    "wind_gusts_10m": (0.0, 180.0),
    "wind_speed_100m": (0.0, 160.0),
    "shortwave_radiation": (0.0, 1100.0),      # W/m²
    "direct_radiation": (0.0, 1000.0),
    "diffuse_radiation": (0.0, 750.0),
    "et0_fao_evapotranspiration": (-0.2, 2.5), # mm/h
}

# Variables categóricas: conjunto de valores válidos.
VALID_CATEGORICAL: dict[str, set[float]] = {
    "is_day": {0.0, 1.0},
    "weather_code": {
        0, 1, 2, 3,             # despejado → cubierto
        45, 48,                 # niebla
        51, 53, 55, 56, 57,     # llovizna
        61, 63, 65, 66, 67,     # lluvia
        71, 73, 75, 77,         # nieve
        80, 81, 82, 85, 86,     # chubascos
        95, 96, 99,             # tormenta
    },
}

# ---- Variables Meteostat ----------------------------------------------------
PLAUSIBLE_RANGES_METEOSTAT: dict[str, tuple[float | None, float | None]] = {
    "temp": (-25.0, 48.0),
    "dwpt": (-35.0, 30.0),
    "rhum": (0.0, 100.0),
    "prcp": (0.0, 120.0),
    "snow": (0.0, 2000.0),     # mm
    "wdir": (0.0, 360.0),
    "wspd": (0.0, 160.0),
    "wpgt": (0.0, 220.0),
    "pres": (975.0, 1050.0),   # presión reducida a nivel del mar
    "tsun": (0.0, 60.0),       # minutos de sol por hora
}


def count_out_of_range(series, lo: float | None, hi: float | None) -> int:
    """Nº de valores no nulos fuera de [lo, hi]."""
    s = series.dropna()
    mask = s.notna() & False
    if lo is not None:
        mask = mask | (s < lo)
    if hi is not None:
        mask = mask | (s > hi)
    return int(mask.sum())
