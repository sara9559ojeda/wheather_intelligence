"""Verificación de fundamentación numérica de la respuesta de la IA.

Claude puede redactar mejor que las plantillas, pero no se le permite introducir
cifras que el sistema no calculó. Antes de mostrar su texto se extraen todos los
números y se comprueba que cada uno aparece (con tolerancia de redondeo) en el
contexto o en la pregunta. Si alguno no aparece, se descarta la respuesta de la
IA y se muestra la determinista.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
SMALL_INT_MAX = 12      # «3 variables», «6 h», «12 meses»: conteos triviales
DECIMAL_TOL = 0.051     # el contexto trae 1 decimal; el texto puede redondear
INT_TOL = 0.51


def _parse(token: str) -> float:
    return float(token.replace(",", "."))


def numbers_in(text: str) -> list[str]:
    return _NUMBER.findall(text)


def _collect(obj, out: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        out.update(_parse(t) for t in numbers_in(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect(k, out)
            _collect(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect(v, out)


def context_numbers(context: dict, extra_text: str = "") -> set[float]:
    out: set[float] = set()
    _collect(context, out)
    _collect(extra_text, out)
    return out


def ungrounded_numbers(answer: str, allowed: set[float]) -> list[str]:
    """Números de ``answer`` que no están respaldados por ``allowed`` (vacío = respuesta válida)."""
    bad: list[str] = []
    for tok in numbers_in(answer):
        x = _parse(tok)
        is_int = "." not in tok and "," not in tok
        if is_int and x <= SMALL_INT_MAX:
            continue
        tol = INT_TOL if is_int else DECIMAL_TOL
        if not any(abs(x - a) <= tol for a in allowed):
            bad.append(tok)
    return bad
