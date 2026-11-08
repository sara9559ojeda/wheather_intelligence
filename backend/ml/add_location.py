"""Orquesta el pipeline completo de UNA ciudad del catálogo (``locations.toml``).

Pasos (cada uno en su propio proceso, con ``WI_PIPELINE_LOCATION=<slug>``):

    1. descarga del histórico ERA5 de Open-Meteo      (acquisition.download_historical)
    2. carga a PostgreSQL                             (etl.load_historical)
    3. limpieza + features + partición + objetivos    (preparation.build_dataset)
    4. entrenamiento de los 5 horizontes              (modeling.run_training)
    5. clustering K-Means                             (mining.run_clustering)
    6. detección de anomalías (Isolation Forest)      (mining.run_anomalies)

Uso::

    python -m backend.ml.add_location --slug bogota
    python -m backend.ml.add_location --all            # todas las del catálogo
    python -m backend.ml.add_location --slug cali --from-step build_dataset

Es idempotente: la descarga reutiliza la caché por año, la carga ignora filas
duplicadas y el entrenamiento reemplaza solo los modelos de esa ciudad. Cuando
termina, la ciudad aparece en ``GET /api/locations`` y en el selector de la app.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time

from backend.ml.config_loader import DEFAULT_SLUG, load_catalog

log = logging.getLogger("add_location")

STEPS: list[tuple[str, str]] = [
    ("download", "backend.ml.acquisition.download_historical"),
    ("load", "backend.app.services.etl.load_historical"),
    ("build_dataset", "backend.ml.preparation.build_dataset"),
    ("train", "backend.ml.modeling.run_training"),
    ("clustering", "backend.ml.mining.run_clustering"),
    ("anomalies", "backend.ml.mining.run_anomalies"),
]
STEP_NAMES = [name for name, _ in STEPS]


def run_pipeline(slug: str, from_step: str = "download") -> None:
    env = {**os.environ, "WI_PIPELINE_LOCATION": slug, "PYTHONIOENCODING": "utf-8"}
    start = STEP_NAMES.index(from_step)
    t_total = time.monotonic()
    for name, module in STEPS[start:]:
        log.info("[%s] ▶ %s", slug, name)
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, "-m", module], env=env, check=False)
        if proc.returncode != 0:
            raise SystemExit(
                f"[{slug}] el paso '{name}' falló (código {proc.returncode}). "
                f"Reanuda con: python -m backend.ml.add_location --slug {slug} --from-step {name}"
            )
        log.info("[%s] ✔ %s en %.0f s", slug, name, time.monotonic() - t0)
    log.info("[%s] pipeline completo en %.1f min", slug, (time.monotonic() - t_total) / 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    catalog = list(load_catalog())
    p = argparse.ArgumentParser(description="Descarga, entrena y publica una ciudad del catálogo.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug", choices=[DEFAULT_SLUG, *catalog], help="Ciudad a procesar.")
    group.add_argument("--all", action="store_true", help="Todas las ciudades del catálogo.")
    p.add_argument("--from-step", choices=STEP_NAMES, default="download",
                   help="Reanuda desde este paso (los anteriores se dan por hechos).")
    args = p.parse_args()

    slugs = catalog if args.all else [args.slug]
    for slug in slugs:
        run_pipeline(slug, args.from_step)


if __name__ == "__main__":
    main()
