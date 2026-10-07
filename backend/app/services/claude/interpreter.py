"""Orquestación de la interpretación con IA (fase 7).

    resumen estructurado (Python)  ->  Claude  ->  texto  ->  ai_analyses

Con caché: si hay un análisis de hace menos de ``ai_cache_minutes`` no se vuelve
a llamar a la API. Con *fallback*: si no hay clave o la API falla, se devuelve
el último análisis disponible (marcado como obsoleto) o solo el resumen.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import AiAnalysis
from backend.app.db.queries import get_location
from backend.app.services.claude.client import ClaudeUnavailable, call_claude, is_configured
from backend.app.services.claude.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from backend.app.services.dashboard import build_ai_summary

log = logging.getLogger(__name__)


def _latest_analysis(session: Session, location_id: int) -> AiAnalysis | None:
    return session.scalar(
        select(AiAnalysis)
        .where(AiAnalysis.location_id == location_id)
        .order_by(AiAnalysis.generated_at.desc())
        .limit(1)
    )


def _serialize(row: AiAnalysis, *, cached: bool) -> dict:
    return {
        "status": "cached" if cached else "ok",
        "interpretation": row.output_text,
        "model": row.model,
        "generated_at": row.generated_at.isoformat(),
        "tokens": {"input": row.input_tokens, "output": row.output_tokens},
        "structured_summary": row.input_summary,
        "cached": cached,
    }


def get_or_create_interpretation(
    session: Session, slug: str, *, force: bool = False
) -> dict:
    """Devuelve la interpretación (de caché o recién generada). No hace commit."""
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    last = _latest_analysis(session, loc.id)
    if last is not None and not force:
        age = datetime.now(UTC) - last.generated_at
        if age < timedelta(minutes=settings.ai_cache_minutes):
            return _serialize(last, cached=True)

    summary = build_ai_summary(session, slug)

    if not is_configured():
        return {
            "status": "no_api_key",
            "note": "Configura ANTHROPIC_API_KEY en el backend para la interpretación.",
            "interpretation": last.output_text if last else None,
            "stale": last is not None,
            "structured_summary": summary,
            "cached": False,
        }

    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        resp = call_claude(SYSTEM_PROMPT, USER_TEMPLATE.format(payload=payload))
    except ClaudeUnavailable as exc:
        log.warning("Interpretación no disponible (%s); se usa el fallback", exc)
        return {
            "status": "unavailable",
            "error": str(exc),
            "interpretation": last.output_text if last else None,
            "stale": last is not None,
            "structured_summary": summary,
            "cached": False,
        }

    row = AiAnalysis(
        location_id=loc.id,
        period=summary.get("period", "current"),
        input_summary=summary,
        model=resp.model,
        output_text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )
    session.add(row)
    session.flush()
    log.info("Interpretación generada (%s, %d+%d tokens)",
             resp.model, resp.input_tokens, resp.output_tokens)
    return {
        "status": "ok",
        "interpretation": resp.text,
        "model": resp.model,
        "generated_at": row.generated_at.isoformat(),
        "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
        "structured_summary": summary,
        "cached": False,
    }


def maybe_generate_on_event(session: Session, slug: str, *, is_anomaly: bool) -> None:
    """Genera una interpretación tras una observación nueva **solo si** hay
    anomalía y no hay un análisis reciente. Mantiene el gasto acotado."""
    if not is_anomaly or not is_configured():
        return
    loc = get_location(session, slug)
    if loc is None:
        return
    last = _latest_analysis(session, loc.id)
    if last is not None:
        age = datetime.now(UTC) - last.generated_at
        if age < timedelta(minutes=settings.ai_min_gap_minutes):
            return
    try:
        get_or_create_interpretation(session, slug, force=True)
    except Exception:  # noqa: BLE001 — nunca debe romper la ingesta
        log.exception("Fallo generando interpretación tras evento")
