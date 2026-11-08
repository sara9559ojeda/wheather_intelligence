"""Analista contextual: responde preguntas usando solo el contexto estructurado."""

from backend.app.services.analyst.service import ask, suggested_questions

__all__ = ["ask", "suggested_questions"]
