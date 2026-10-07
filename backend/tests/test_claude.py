"""Pruebas de la integración con Claude (fase 7).

No se llama a la API real: se sustituye ``call_claude`` por un doble.
Las que tocan la BD llevan el marcador ``db``.
"""

from __future__ import annotations

import pytest

from backend.app.services.claude import client as claude_client
from backend.app.services.claude import interpreter

pytestmark = pytest.mark.db

_FAKE_TEXT = (
    "**Resumen**\nHace 21 °C.\n**Patrones**\nCiclo diario.\n"
    "**Comparación histórica**\nPercentil 80.\n**Anomalías**\nNinguna.\n"
    "**Predicción**\n22 °C ± 2.\n**Interpretación**\nNormal.\n"
    "**Recomendaciones de observación**\nNada especial."
)


class _FakeResponse:
    text = _FAKE_TEXT
    model = "claude-haiku-4-5"
    input_tokens = 850
    output_tokens = 240


@pytest.fixture
def _no_key(monkeypatch):
    monkeypatch.setattr(interpreter, "is_configured", lambda: False)


@pytest.fixture
def _fake_claude(monkeypatch):
    monkeypatch.setattr(interpreter, "is_configured", lambda: True)
    calls = []

    def _fake(system, user_message, *, max_tokens=1200):
        calls.append(user_message)
        return _FakeResponse()

    monkeypatch.setattr(interpreter, "call_claude", _fake)
    return calls


def _clean(session):
    from backend.app.db.models import AiAnalysis

    session.query(AiAnalysis).delete()
    session.flush()


def test_is_configured_reads_settings(monkeypatch):
    monkeypatch.setattr(claude_client.settings, "anthropic_api_key", "  ")
    assert claude_client.is_configured() is False
    monkeypatch.setattr(claude_client.settings, "anthropic_api_key", "sk-ant-xxx")
    assert claude_client.is_configured() is True


def test_fallback_without_api_key(_no_key):
    from backend.app.db.session import session_scope

    with session_scope() as s:
        result = interpreter.get_or_create_interpretation(s, "madrid_barajas")
    assert result["status"] == "no_api_key"
    assert "structured_summary" in result
    assert set(result["structured_summary"]) >= {"current", "regime", "anomaly", "prediction"}


def test_generates_persists_and_caches(_fake_claude):
    from backend.app.db.models import AiAnalysis
    from backend.app.db.session import session_scope

    with session_scope() as s:
        _clean(s)

    with session_scope() as s:
        r1 = interpreter.get_or_create_interpretation(s, "madrid_barajas")
    assert r1["status"] == "ok"
    assert r1["interpretation"] == _FAKE_TEXT
    assert r1["tokens"] == {"input": 850, "output": 240}
    assert len(_fake_claude) == 1

    with session_scope() as s:
        n = s.query(AiAnalysis).count()
    assert n == 1

    # segunda llamada dentro de la ventana de caché -> no vuelve a llamar
    with session_scope() as s:
        r2 = interpreter.get_or_create_interpretation(s, "madrid_barajas")
    assert r2["status"] == "cached"
    assert len(_fake_claude) == 1

    # force -> ignora la caché
    with session_scope() as s:
        r3 = interpreter.get_or_create_interpretation(s, "madrid_barajas", force=True)
    assert r3["status"] == "ok"
    assert len(_fake_claude) == 2

    with session_scope() as s:
        _clean(s)


def test_unavailable_falls_back_to_last(_fake_claude, monkeypatch):
    from backend.app.db.session import session_scope

    with session_scope() as s:
        _clean(s)
        interpreter.get_or_create_interpretation(s, "madrid_barajas")  # crea uno

    def _boom(*a, **k):
        raise interpreter.ClaudeUnavailable("API caída")

    monkeypatch.setattr(interpreter, "call_claude", _boom)
    with session_scope() as s:
        r = interpreter.get_or_create_interpretation(s, "madrid_barajas", force=True)
    assert r["status"] == "unavailable"
    assert r["interpretation"] == _FAKE_TEXT  # el último disponible
    assert r["stale"] is True

    with session_scope() as s:
        _clean(s)


def test_maybe_generate_only_on_anomaly(_fake_claude):
    from backend.app.db.session import session_scope

    with session_scope() as s:
        _clean(s)
        interpreter.maybe_generate_on_event(s, "madrid_barajas", is_anomaly=False)
    assert len(_fake_claude) == 0

    with session_scope() as s:
        interpreter.maybe_generate_on_event(s, "madrid_barajas", is_anomaly=True)
    assert len(_fake_claude) == 1

    with session_scope() as s:
        _clean(s)
