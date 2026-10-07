"""Partición **cronológica** del dataset (fase 1, §15).

Nada de particiones aleatorias: los datos son una serie temporal. Se corta por
orden de tiempo en 70 % / 15 % / 15 % y se deja un **hueco** (``gap_hours``)
entre tramos igual al horizonte máximo de predicción, para que los *targets*
del final de un tramo (temperatura hasta 24 h en el futuro) no caigan dentro
del tramo siguiente.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.app.services.features.builder import HORIZONS_HOURS

log = logging.getLogger(__name__)

TS = "timestamp"
DEFAULT_GAP_HOURS = max(HORIZONS_HOURS)


@dataclass
class SplitResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict

    def log(self) -> None:
        for name in ("train", "valid", "test"):
            part = getattr(self, name)
            b = self.boundaries[name]
            log.info("  %-5s %7d filas  %s → %s", name, len(part), b["start"], b["end"])


def chronological_split(
    df: pd.DataFrame,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    gap_hours: int = DEFAULT_GAP_HOURS,
) -> SplitResult:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"Las fracciones deben sumar 1.0 (suman {sum(fractions)}).")

    df = df.sort_values(TS).reset_index(drop=True)
    n = len(df)
    n_train = int(n * fractions[0])
    n_valid = int(n * fractions[1])

    train = df.iloc[:n_train]
    valid = df.iloc[n_train + gap_hours : n_train + gap_hours + n_valid]
    test = df.iloc[n_train + gap_hours + n_valid + gap_hours :]

    def _bounds(part: pd.DataFrame) -> dict:
        return {
            "start": part[TS].min().isoformat(),
            "end": part[TS].max().isoformat(),
            "n_rows": int(len(part)),
        }

    boundaries = {
        "gap_hours": gap_hours,
        "fractions": list(fractions),
        "total_rows": n,
        "train": _bounds(train),
        "valid": _bounds(valid),
        "test": _bounds(test),
    }
    return SplitResult(
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
        boundaries,
    )


def write_split(result: SplitResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        getattr(result, name).to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "split_meta.json").write_text(
        json.dumps(result.boundaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
