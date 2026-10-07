"""CLI: construye el dataset listo para modelar (fase 3).

    observaciones (PostgreSQL)
        -> limpieza
        -> features derivadas (lags, medias móviles, cíclicas, deltas)
        -> targets (temperatura en t+H)
        -> partición cronológica 70/15/15 con hueco
        -> data/processed/{train,valid,test}.parquet + split_meta.json

Uso:
    python -m backend.ml.preparation.build_dataset
"""

from __future__ import annotations

import json
import logging

from backend.app.db.queries import load_observations_df
from backend.app.db.session import session_scope
from backend.app.services.features.builder import (
    HORIZONS_HOURS,
    add_targets,
    build_feature_frame,
    list_feature_columns,
    target_column,
)
from backend.ml.config_loader import PROCESSED_DIR, ensure_data_dirs, load_config
from backend.ml.preparation.clean import clean_observations
from backend.ml.preparation.split import chronological_split, write_split

log = logging.getLogger("preparation.build_dataset")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config()
    ensure_data_dirs()

    with session_scope() as session:
        raw = load_observations_df(session, cfg.slug, sources=("historical",))
    log.info("Observaciones cargadas de la BD: %d filas", len(raw))

    clean, creport = clean_observations(raw)
    creport.log()

    feats = build_feature_frame(clean)
    feature_cols = list_feature_columns(feats)
    log.info("Tras features: %d filas, %d columnas (%d features del modelo)",
             len(feats), feats.shape[1], len(feature_cols))

    feats = add_targets(feats, HORIZONS_HOURS)

    split = chronological_split(feats)
    log.info("Partición cronológica (hueco = %d h):", split.boundaries["gap_hours"])
    split.log()

    write_split(split, PROCESSED_DIR)

    # metadatos del dataset procesado (versionable en git)
    manifest = {
        "location": cfg.name,
        "n_rows_raw": int(len(raw)),
        "n_rows_after_cleaning": int(creport.n_rows_out),
        "n_rows_modelling": int(len(feats)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_columns": [target_column(h) for h in HORIZONS_HOURS],
        "horizons_hours": list(HORIZONS_HOURS),
        "cleaning": {
            "duplicates_removed": creport.duplicates_removed,
            "clipped": {k: v for k, v in creport.clipped.items() if v},
        },
        "split": split.boundaries,
    }
    (PROCESSED_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Escrito %s y dataset_manifest.json", PROCESSED_DIR)


if __name__ == "__main__":
    main()
