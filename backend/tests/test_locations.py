"""Catálogo de ciudades: configuración por slug, rutas de artefactos y API ``?location=``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_current_user
from backend.app.main import app
from backend.ml.config_loader import (
    DEFAULT_SLUG,
    PROCESSED_DIR,
    load_catalog,
    load_config,
    model_artifact_name,
    processed_dir,
)


def test_catalog_has_colombian_cities():
    catalog = load_catalog()
    assert {"bogota", "medellin", "cali", "barranquilla", "cartagena"} <= set(catalog)


def test_default_config_is_madrid():
    cfg = load_config()
    assert cfg.slug == DEFAULT_SLUG == "madrid_barajas"
    assert cfg.timezone == "Europe/Madrid"


def test_catalog_city_overlays_shared_settings():
    bogota = load_config(slug="bogota")
    madrid = load_config(slug=DEFAULT_SLUG)
    assert bogota.name == "Bogotá"
    assert bogota.timezone == "America/Bogota"
    assert bogota.country == "CO"
    assert (bogota.latitude, bogota.longitude) == (4.711, -74.0721)
    # Lo común (variables, API, reintentos) se hereda de location.toml.
    assert bogota.hourly_variables == madrid.hourly_variables
    assert bogota.archive_base_url == madrid.archive_base_url
    # Sin estación Meteostat fuera de Madrid.
    assert not bogota.meteostat_station_id


def test_unknown_city_is_rejected():
    with pytest.raises(KeyError):
        load_config(slug="atlantida")


def test_artifact_and_processed_paths_keep_madrid_legacy_names():
    assert model_artifact_name(6, DEFAULT_SLUG) == "model_temp_h6.joblib"
    assert model_artifact_name(6, "cali") == "model_cali_temp_h6.joblib"
    assert processed_dir(DEFAULT_SLUG) == PROCESSED_DIR
    assert processed_dir("cali") == PROCESSED_DIR / "cali"


@pytest.mark.db
class TestLocationsApi:
    @pytest.fixture(autouse=True)
    def _logged_in(self):
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=0)
        yield
        app.dependency_overrides.pop(get_current_user, None)

    client = TestClient(app)

    def test_lists_ready_locations_default_first(self):
        r = self.client.get("/api/locations")
        assert r.status_code == 200
        rows = r.json()
        assert rows[0]["slug"] == DEFAULT_SLUG
        assert rows[0]["is_default"] is True
        assert all(not row["is_default"] for row in rows[1:])
        assert {"slug", "name", "latitude", "longitude", "timezone", "country"} <= set(rows[0])

    def test_unknown_location_is_404(self):
        r = self.client.get("/api/weather/current", params={"location": "atlantida"})
        assert r.status_code == 404

    def test_invalid_slug_is_rejected(self):
        r = self.client.get("/api/weather/current", params={"location": "../etc"})
        assert r.status_code == 422

    def test_locations_requires_session(self):
        app.dependency_overrides.pop(get_current_user, None)
        assert self.client.get("/api/locations").status_code == 401
