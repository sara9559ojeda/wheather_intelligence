"""Consultas reutilizables sobre la base de datos."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Location, WeatherObservation

_OBSERVATION_COLUMNS = [
    "observed_at",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "surface_pressure", "pressure_msl", "precipitation", "rain", "snowfall",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "wind_speed_100m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "et0_fao_evapotranspiration", "weather_code", "is_day",
]

_NUMERIC = [c for c in _OBSERVATION_COLUMNS if c not in ("observed_at", "is_day", "weather_code")]


def get_location(session: Session, slug: str) -> Location | None:
    return session.scalar(select(Location).where(Location.slug == slug))


def load_observations_df(
    session: Session,
    slug: str,
    *,
    sources: tuple[str, ...] | None = None,
    with_id: bool = False,
) -> pd.DataFrame:
    """Observaciones de una ubicación como DataFrame ordenado por tiempo.

    La columna ``observed_at`` se renombra a ``timestamp`` (convención de
    ``features.builder``). Los ``Numeric`` de PostgreSQL se convierten a float64.
    Con ``with_id=True`` se incluye la columna ``observation_id`` (necesaria para
    persistir clusters y anomalías, que referencian ``weather_observations.id``).
    """
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"No existe la ubicación '{slug}' en la base de datos.")

    names = (["observation_id"] if with_id else []) + _OBSERVATION_COLUMNS
    select_cols = ([WeatherObservation.id] if with_id else []) + [
        getattr(WeatherObservation, c) for c in _OBSERVATION_COLUMNS
    ]
    stmt = select(*select_cols).where(WeatherObservation.location_id == loc.id)
    if sources:
        stmt = stmt.where(WeatherObservation.source.in_(sources))
    stmt = stmt.order_by(WeatherObservation.observed_at)

    df = pd.DataFrame(session.execute(stmt).all(), columns=names)
    df = df.rename(columns={"observed_at": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df[_NUMERIC] = df[_NUMERIC].astype("float64")
    if "is_day" in df:
        df["is_day"] = df["is_day"].astype("Float64").astype("float64")
    if "weather_code" in df:
        df["weather_code"] = df["weather_code"].astype("Int64")
    if with_id:
        df["observation_id"] = df["observation_id"].astype("int64")
    return df
