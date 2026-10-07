"""Pruebas de la API (fase 6).

Se usa ``TestClient`` **sin** el context manager → no se ejecuta el lifespan
(nada de scheduler ni de llamadas a Open-Meteo). Requieren la base de datos
poblada (marcador ``db``):  pytest -m db
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

pytestmark = pytest.mark.db
client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_meta_reports_observation_counts():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["observations"]["historical"] > 50_000
    assert body["location"] == "Madrid-Barajas"


def test_current_weather():
    r = client.get("/api/weather/current")
    assert r.status_code == 200
    body = r.json()
    assert body["observation"]["temperature_2m"] is not None
    assert body["age_minutes"] >= 0


def test_history_returns_ordered_points():
    r = client.get("/api/weather/history", params={"hours": 72})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    times = [row["observed_at"] for row in rows]
    assert times == sorted(times)


def test_stats_has_temperature():
    r = client.get("/api/stats", params={"days": 365})
    assert r.status_code == 200
    variables = {v["variable"] for v in r.json()["variables"]}
    assert "temperature_2m" in variables


def test_anomalies_only_flagged():
    r = client.get("/api/anomalies", params={"limit": 20, "only_flagged": True})
    assert r.status_code == 200
    assert all(a["is_anomaly"] for a in r.json())


def test_clusters():
    r = client.get("/api/clusters")
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 3
    assert len(body["clusters"]) == 3


def test_model_runs_and_performance():
    # >= 25: la fase 5 crea 5 horizontes x (2 baselines + 3 modelos); cada
    # reentrenamiento (fase 9) añade más filas (retadores, promocionados o no).
    runs = client.get("/api/model/runs").json()
    assert len(runs) >= 25
    assert {r["horizon_hours"] for r in runs} == {1, 3, 6, 12, 24}
    perf = client.get("/api/model/performance").json()
    assert {m["horizon_hours"] for m in perf} == {1, 3, 6, 12, 24}
    assert all(not m["model_type"].startswith("baseline") for m in perf)


def test_predictions_latest():
    r = client.get("/api/predictions/latest")
    assert r.status_code == 200
    preds = r.json()
    assert {p["horizon_hours"] for p in preds} == {1, 3, 6, 12, 24}
    for p in preds:
        assert p["lower_bound"] <= p["predicted_temperature"] <= p["upper_bound"]


def test_comparison_latest_has_conditioned_percentiles():
    r = client.get("/api/comparison/latest")
    assert r.status_code == 200
    body = r.json()
    variables = {p["variable"] for p in body["percentiles"]}
    assert "temperature_2m" in variables
    for p in body["percentiles"]:
        assert 0 <= p["percentile"] <= 100


def test_dashboard_bundles_everything():
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    body = r.json()
    for key in ("current", "predictions", "historical_comparison",
                "recent_anomalies", "clusters", "model_performance"):
        assert key in body


def test_insights_returns_structured_summary():
    r = client.get("/api/insights")
    assert r.status_code == 200
    body = r.json()
    # sin ANTHROPIC_API_KEY en el entorno de test -> fallback con el resumen
    assert body["status"] in {"no_api_key", "ok", "cached", "unavailable"}
    if body["status"] == "no_api_key":
        assert "structured_summary" in body
        assert "current" in body["structured_summary"]


def test_openapi_schema_available():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"] == "Weather Intelligence API"
