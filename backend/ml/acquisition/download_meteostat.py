"""CLI: descarga las observaciones reales de estación (Meteostat) para contraste.

Uso:
    python -m backend.ml.acquisition.download_meteostat

Produce:
    data/raw/meteostat_<station_id>_hourly.parquet
    data/raw/meteostat_<station_id>_hourly.meta.json
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from backend.ml.acquisition.metadata import build_metadata, write_metadata
from backend.ml.acquisition.meteostat_station import (
    METEOSTAT_COLUMNS,
    fetch_station_hourly,
    fetch_station_metadata,
)
from backend.ml.config_loader import RAW_DIR, ensure_data_dirs, load_config

log = logging.getLogger("acquisition.meteostat")


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Descarga histórico horario de una estación Meteostat.")
    p.add_argument("--station", default=cfg.meteostat_station_id,
                   help=f"ID de estación Meteostat. Def: {cfg.meteostat_station_id}")
    p.add_argument("--start", type=date.fromisoformat, default=cfg.historical_start_date)
    p.add_argument("--end", type=date.fromisoformat, default=cfg.historical_end_date)
    return p.parse_args()


_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    args = parse_args()
    ensure_data_dirs()

    station_meta = fetch_station_metadata(args.station)
    station_name = station_meta.get("name", {}).get("en", "?")
    log.info("Estación %s: %s (%s)", args.station, station_name, station_meta.get("country"))

    df = fetch_station_hourly(args.station, args.start, args.end)
    log.info("Descargadas %d filas, %d columnas", len(df), df.shape[1])

    out_path = RAW_DIR / f"meteostat_{args.station}_hourly.parquet"
    df.to_parquet(out_path, index=False)

    meta = build_metadata(
        data_path=out_path,
        df=df,
        timestamp_column="timestamp",
        source=f"Meteostat — estación {args.station} ({station_name})",
        source_url="https://dev.meteostat.net/",
        license_name="CC BY 4.0",
        extra={
            "station": {
                "id": args.station,
                "identifiers": station_meta.get("identifiers", {}),
                "location": station_meta.get("location", {}),
                "timezone": station_meta.get("timezone"),
            },
            "column_glossary": METEOSTAT_COLUMNS,
            "notes": (
                "Observación real de estación (METAR/SYNOP agregados). Presión = nivel "
                "del mar (pres). Espere huecos y periodos sin datos, sobre todo antes de ~2005."
            ),
        },
    )
    meta_path = write_metadata(meta, out_path)

    log.info("OK  →  %s", out_path)
    log.info("OK  →  %s", meta_path)
    log.info("Rango real: %s … %s", meta["date_range_utc"]["start"], meta["date_range_utc"]["end"])
    completeness = 100 * meta["n_rows"] / max(meta["expected_hourly_rows"], 1)
    log.info("Completitud aproximada: %.1f%% (%d de %d filas horarias esperadas)",
             completeness, meta["n_rows"], meta["expected_hourly_rows"])


if __name__ == "__main__":
    main()
