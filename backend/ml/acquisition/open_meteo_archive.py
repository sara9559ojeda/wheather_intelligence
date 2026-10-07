"""Cliente de la Open-Meteo Historical Weather API (reanálisis ERA5).

Se usa una sola vez para construir el dataset histórico estático del proyecto
(ver ``docs/fase-1-analisis-y-arquitectura.md`` §7). No requiere API key.
Licencia de los datos: CC BY 4.0 (atribución obligatoria).

Estrategia de descarga:
  - una petición por año natural (payload manejable, ~0.9 MB / 18 s por año);
  - formato CSV (más compacto que JSON);
  - caché por año en ``data/interim/acquisition_cache/`` → re-ejecutar es barato;
  - reintentos con backoff exponencial ante fallos de red o 5xx.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

# Errores transitorios que justifican reintentar.
_RETRYABLE = (requests.ConnectionError, requests.Timeout, requests.HTTPError)


class RateLimited(requests.HTTPError):
    """HTTP 429 — límite de peticiones de Open-Meteo alcanzado."""


class OpenMeteoError(RuntimeError):
    """La API respondió con un error de negocio (``{"error": true, ...}``)."""


class OpenMeteoArchiveClient:
    """Descarga series horarias históricas de una única ubicación."""

    def __init__(
        self,
        *,
        base_url: str,
        latitude: float,
        longitude: float,
        variables: list[str],
        timeout_seconds: int = 180,
        retry_attempts: int = 4,
        polite_delay_seconds: float = 1.0,
        cache_dir: Path | None = None,
    ) -> None:
        self.base_url = base_url
        self.latitude = latitude
        self.longitude = longitude
        self.variables = list(variables)
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.polite_delay_seconds = polite_delay_seconds
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "weather-intelligence/0.1 (academic project)"})
        # metadatos de la celda de rejilla ERA5 (los rellena la primera descarga)
        self.grid_metadata: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def _fetch_year_raw(self, year: int, start_iso: str, end_iso: str) -> str:
        """Devuelve el texto CSV del tramo [start_iso, end_iso]. Reintenta ante fallos."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=4, min=15, max=120),
            retry=retry_if_exception_type(_RETRYABLE),
            before_sleep=lambda rs: log.warning(
                "Reintento %d para el año %d (%s)",
                rs.attempt_number,
                year,
                rs.outcome.exception() if rs.outcome else "?",
            ),
        )
        def _do_request() -> str:
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": start_iso,
                "end_date": end_iso,
                "hourly": ",".join(self.variables),
                "timezone": "UTC",
                "format": "csv",
            }
            resp = self._session.get(self.base_url, params=params, timeout=self.timeout_seconds)
            # Open-Meteo devuelve 400 + JSON en errores de negocio.
            ctype = resp.headers.get("content-type", "")
            if resp.status_code == 400 and "application/json" in ctype:
                raise OpenMeteoError(resp.json().get("reason", resp.text))
            if resp.status_code == 429:
                # Límite por minuto: respeta Retry-After si viene, si no espera ~65 s.
                delay = int(resp.headers.get("Retry-After", "65"))
                log.warning("HTTP 429 (año %d); espera %d s antes de reintentar", year, delay)
                time.sleep(delay)
                raise RateLimited("429 Too Many Requests", response=resp)
            resp.raise_for_status()
            return resp.text

        return _do_request()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_csv(text: str) -> pd.DataFrame:
        """CSV de Open-Meteo → DataFrame con nombres de columna canónicos.

        Estructura del CSV:
            línea 1  cabecera de metadatos (latitude,longitude,elevation,...)
            línea 2  valores de metadatos
            línea 3  vacía
            línea 4  cabecera de datos  (time,temperature_2m (°C),...)
            línea 5+ datos
        """
        meta_df = pd.read_csv(io.StringIO(text), nrows=1)
        data_df = pd.read_csv(io.StringIO(text), skiprows=3)

        # "temperature_2m (°C)" -> "temperature_2m"
        data_df.columns = [c.split(" (")[0].strip() for c in data_df.columns]
        data_df = data_df.rename(columns={"time": "timestamp"})
        data_df["timestamp"] = pd.to_datetime(data_df["timestamp"], utc=True)

        data_df.attrs["grid_latitude"] = float(meta_df["latitude"].iloc[0])
        data_df.attrs["grid_longitude"] = float(meta_df["longitude"].iloc[0])
        data_df.attrs["grid_elevation_m"] = float(meta_df["elevation"].iloc[0])
        return data_df

    # ------------------------------------------------------------------ #
    def _year(self, year: int, period_start: date, period_end: date) -> pd.DataFrame:
        """Un año (recortado al periodo pedido), usando caché si el año es completo."""
        y_start = max(date(year, 1, 1), period_start)
        y_end = min(date(year, 12, 31), period_end)
        is_full_year = y_start == date(year, 1, 1) and y_end == date(year, 12, 31)

        # Solo se cachea un año natural completo (un año parcial cambiaría cada día).
        cache_file = (
            self.cache_dir / f"open_meteo_{year}.parquet"
            if (self.cache_dir and is_full_year)
            else None
        )
        if cache_file and cache_file.exists():
            log.info("Año %d: caché → %s", year, cache_file.name)
            return pd.read_parquet(cache_file)

        log.info("Año %d: descargando de Open-Meteo (%s … %s)…", year, y_start, y_end)
        df = self._parse_csv(
            self._fetch_year_raw(year, y_start.isoformat(), y_end.isoformat())
        )
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file, index=False)
        time.sleep(self.polite_delay_seconds)
        return df

    # ------------------------------------------------------------------ #
    def fetch_hourly(self, start_date, end_date) -> pd.DataFrame:
        """Descarga el rango [start_date, end_date] (ambos ``datetime.date``).

        Devuelve un DataFrame horario ordenado, sin duplicados de timestamp,
        recortado exactamente al rango pedido.
        """
        frames: list[pd.DataFrame] = []
        for year in range(start_date.year, end_date.year + 1):
            df_year = self._year(year, start_date, end_date)
            frames.append(df_year)
            if not self.grid_metadata and df_year.attrs.get("grid_latitude") is not None:
                self.grid_metadata = {
                    "grid_latitude": df_year.attrs["grid_latitude"],
                    "grid_longitude": df_year.attrs["grid_longitude"],
                    "grid_elevation_m": df_year.attrs["grid_elevation_m"],
                }

        full = pd.concat(frames, ignore_index=True)
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)
        full = full[(full["timestamp"] >= start_ts) & (full["timestamp"] <= end_ts)]
        full = (
            full.drop_duplicates(subset="timestamp", keep="first")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return full
