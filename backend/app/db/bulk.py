"""Inserción masiva idempotente, respetando el límite de parámetros de PostgreSQL."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

_PG_MAX_PARAMS = 65535


def bulk_upsert(
    session: Session,
    model: Any,
    records: list[dict[str, Any]],
    *,
    conflict_index: list[str] | None = None,
    chunk_size: int = 5_000,
    label: str = "filas",
) -> int:
    """Inserta ``records`` en ``model``. Con ``conflict_index`` usa
    ``ON CONFLICT DO NOTHING`` sobre esas columnas (idempotente).

    Devuelve el nº de registros procesados (no necesariamente insertados).
    """
    if not records:
        return 0
    n_cols = max(len(r) for r in records)
    safe = max(1, min(chunk_size, _PG_MAX_PARAMS // n_cols))

    for start in range(0, len(records), safe):
        chunk = records[start : start + safe]
        stmt = pg_insert(model).values(chunk)
        if conflict_index:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_index)
        session.execute(stmt)
        end = min(start + safe, len(records))
        if end % (safe * 10) < safe or end == len(records):
            log.info("  %s: %d / %d", label, end, len(records))
    return len(records)
