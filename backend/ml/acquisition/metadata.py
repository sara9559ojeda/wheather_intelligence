"""Escritura de *sidecars* de metadatos para cada archivo de datos crudo.

Cada dataset en ``data/raw/`` va acompañado de un ``<nombre>.meta.json`` que
documenta procedencia, rango temporal, nº de filas y un ``sha256`` del archivo.
Los ``.meta.json`` SÍ se versionan en git (los datos no) → trazabilidad.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hexadecimal del contenido de un archivo."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(
    *,
    data_path: Path,
    df: pd.DataFrame,
    timestamp_column: str,
    source: str,
    source_url: str,
    license_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye el diccionario de metadatos de un dataset."""
    ts = pd.to_datetime(df[timestamp_column], utc=True)
    missing_per_column = {
        col: int(df[col].isna().sum()) for col in df.columns
    }
    meta: dict[str, Any] = {
        "dataset_file": data_path.name,
        "source": source,
        "source_url": source_url,
        "license": license_name,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "timestamp_column": timestamp_column,
        "date_range_utc": {
            "start": ts.min().isoformat(),
            "end": ts.max().isoformat(),
        },
        "expected_hourly_rows": int(
            (ts.max() - ts.min()) / pd.Timedelta(hours=1) + 1
        ),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "missing_values_per_column": missing_per_column,
        "sha256": sha256_file(data_path),
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata(meta: dict[str, Any], data_path: Path) -> Path:
    """Escribe ``<data_path sin extensión>.meta.json`` y devuelve su ruta."""
    meta_path = data_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path
