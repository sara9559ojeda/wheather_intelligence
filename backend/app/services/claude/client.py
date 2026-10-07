"""Cliente fino de la API de Anthropic (Claude).

La clave (`ANTHROPIC_API_KEY`) vive solo en el backend (`.env`). Si no está
configurada, ``is_configured()`` devuelve False y la capa superior usa el
*fallback* (no llama a la API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

from backend.app.core.config import settings

log = logging.getLogger(__name__)


class ClaudeUnavailable(RuntimeError):
    """La API de Claude no está disponible (sin clave, error de red o de la API)."""


@dataclass
class ClaudeResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


def is_configured() -> bool:
    return bool(settings.anthropic_api_key.strip())


def call_claude(system: str, user_message: str, *, max_tokens: int = 1200) -> ClaudeResponse:
    if not is_configured():
        raise ClaudeUnavailable("ANTHROPIC_API_KEY no configurada")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=45.0)
    try:
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIStatusError as exc:
        log.warning("Claude API status error %s: %s", exc.status_code, exc.message)
        raise ClaudeUnavailable(f"API error {exc.status_code}") from exc
    except anthropic.APIConnectionError as exc:
        log.warning("Claude API connection error: %s", exc)
        raise ClaudeUnavailable("error de conexión") from exc

    if resp.stop_reason == "refusal":
        raise ClaudeUnavailable("la respuesta fue rechazada por seguridad")

    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    if not text:
        raise ClaudeUnavailable("respuesta vacía")

    return ClaudeResponse(
        text=text,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
