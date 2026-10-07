"""Capa de interpretación con IA generativa (Claude / Anthropic).

Claude NO es el modelo de ML: interpreta en lenguaje natural el resumen
estructurado que produce Python. La clave vive solo en el backend.
"""

from backend.app.services.claude.interpreter import (
    get_or_create_interpretation,
    maybe_generate_on_event,
)

__all__ = ["get_or_create_interpretation", "maybe_generate_on_event"]
