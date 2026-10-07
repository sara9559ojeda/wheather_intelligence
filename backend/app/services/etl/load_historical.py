"""CLI: carga el dataset histórico de ``data/raw`` a PostgreSQL.

Requisitos previos:
    docker compose up -d db
    (cd backend && alembic upgrade head)
    python -m backend.ml.acquisition.download_historical

Uso:
    python -m backend.app.services.etl.load_historical
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import func, select

from backend.app.db.models import SOURCE_HISTORICAL, WeatherObservation
from backend.app.db.session import session_scope
from backend.app.services.etl.historical_loader import load_dataframe, validate
from backend.app.services.etl.location_seed import get_or_create_location
from backend.ml.config_loader import RAW_DIR, load_config

log = logging.getLogger("etl.load_historical")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carga el histórico ERA5 a weather_observations.")
    p.add_argument("--chunk-size", type=int, default=2_000,
                   help="Filas por INSERT (se ajusta si supera el máx. de params de PG).")
    p.add_argument("--fail-on-issues", action="store_true",
                   help="Aborta si la validación encuentra problemas (por defecto solo avisa).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    cfg = load_config()

    parquet = RAW_DIR / f"open_meteo_{cfg.slug}_hourly.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"No existe {parquet}. Ejecuta antes "
            "python -m backend.ml.acquisition.download_historical"
        )

    df = pd.read_parquet(parquet)
    log.info("Leído %s: %d filas, %d columnas", parquet.name, len(df), df.shape[1])

    report = validate(df)
    report.log()
    if args.fail_on_issues and report.total_issues:
        raise SystemExit(
            f"Validación con {report.total_issues} problemas; abortando (--fail-on-issues)."
        )

    def _count(session, location_id: int) -> int:
        return session.scalar(
            select(func.count())
            .select_from(WeatherObservation)
            .where(
                WeatherObservation.location_id == location_id,
                WeatherObservation.source == SOURCE_HISTORICAL,
            )
        )

    with session_scope() as session:
        location = get_or_create_location(session)
        before = _count(session, location.id)
        processed = load_dataframe(
            session, df, location_id=location.id, chunk_size=args.chunk_size
        )
        session.flush()
        after = _count(session, location.id)

    log.info("Hecho. Procesadas %d filas | nuevas: %d | duplicadas (ya existían): %d",
             processed, after - before, processed - (after - before))
    log.info("Total histórico para %s: %d", cfg.name, after)


if __name__ == "__main__":
    main()
