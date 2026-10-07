"""CLI: descarga el dataset histórico principal (Open-Meteo / ERA5).

Uso:
    python -m backend.ml.acquisition.download_historical
    python -m backend.ml.acquisition.download_historical --start 2010-01-01 --no-cache

Produce:
    data/raw/open_meteo_<slug>_hourly.parquet
    data/raw/open_meteo_<slug>_hourly.meta.json
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from backend.ml.acquisition.metadata import build_metadata, write_metadata
from backend.ml.acquisition.open_meteo_archive import OpenMeteoArchiveClient
from backend.ml.config_loader import CACHE_DIR, RAW_DIR, ensure_data_dirs, load_config

log = logging.getLogger("acquisition.historical")


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Descarga histórico horario de Open-Meteo (ERA5).")
    p.add_argument("--start", type=date.fromisoformat, default=cfg.historical_start_date,
                   help=f"Fecha inicial (YYYY-MM-DD). Def: {cfg.historical_start_date}")
    p.add_argument("--end", type=date.fromisoformat, default=cfg.historical_end_date,
                   help=f"Fecha final (YYYY-MM-DD). Def: hoy - {cfg.era5_lag_days}d "
                        f"= {cfg.historical_end_date}")
    p.add_argument("--no-cache", action="store_true", help="Ignora la caché por año.")
    return p.parse_args()


_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    args = parse_args()
    cfg = load_config()
    ensure_data_dirs()

    client = OpenMeteoArchiveClient(
        base_url=cfg.archive_base_url,
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        variables=list(cfg.hourly_variables),
        timeout_seconds=cfg.request_timeout_seconds,
        retry_attempts=cfg.retry_attempts,
        polite_delay_seconds=cfg.polite_delay_seconds,
        cache_dir=None if args.no_cache else CACHE_DIR,
    )

    log.info("Ubicación: %s (%.4f, %.4f)  rango: %s → %s",
             cfg.name, cfg.latitude, cfg.longitude, args.start, args.end)
    df = client.fetch_hourly(args.start, args.end)
    log.info("Descargadas %d filas, %d columnas", len(df), df.shape[1])

    out_path = RAW_DIR / f"open_meteo_{cfg.slug}_hourly.parquet"
    df.to_parquet(out_path, index=False)

    meta = build_metadata(
        data_path=out_path,
        df=df,
        timestamp_column="timestamp",
        source="Open-Meteo Historical Weather API (reanálisis ERA5, ECMWF/Copernicus)",
        source_url=cfg.archive_base_url,
        license_name="CC BY 4.0",
        extra={
            "location": {
                "name": cfg.name,
                "requested_latitude": cfg.latitude,
                "requested_longitude": cfg.longitude,
                "timezone": cfg.timezone,
                **client.grid_metadata,
            },
            "hourly_variables_requested": list(cfg.hourly_variables),
            "notes": (
                "ERA5 es reanálisis, no observación directa. Open-Meteo ajusta a la "
                "celda de rejilla más cercana (ver location.grid_*). Retraso de ~5-7 días."
            ),
        },
    )
    meta_path = write_metadata(meta, out_path)

    log.info("OK  →  %s", out_path)
    log.info("OK  →  %s", meta_path)
    log.info("Rango real: %s … %s | duplicados: %d | filas esperadas: %d",
             meta["date_range_utc"]["start"], meta["date_range_utc"]["end"],
             meta["duplicate_timestamps"], meta["expected_hourly_rows"])


if __name__ == "__main__":
    main()
