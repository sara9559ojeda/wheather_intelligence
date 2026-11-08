"""F1 — ¿Qué tan normal es el tiempo de hoy?

Percentiles, ventanas de referencia, estados sin datos y ausencia de fugas del futuro.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.db.session import SessionLocal
from backend.app.services.analysis import normality as nm
from backend.tests.conftest import make_city_rows, utc

# ---------------------------------------------------------------- sin BD


def test_percentile_is_share_less_or_equal():
    vals = np.sort(np.array([1.0, 2.0, 2.0, 3.0, 10.0]))
    assert nm.percentile_of(vals, 2.0) == 60.0     # 1, 2, 2 <= 2
    assert nm.percentile_of(vals, 0.5) == 0.0
    assert nm.percentile_of(vals, 10.0) == 100.0


def test_percentile_handles_ties_at_float64():
    # 0,1 mm debe ser igual a sí mismo (con float32 dejaba de serlo)
    vals = np.sort(np.array([0.1] * 5 + [0.5] * 5, dtype="float64"))
    assert nm.percentile_of(vals, 0.1) == 50.0


@pytest.mark.parametrize("p,key", [
    (1.0, "exceptional_low"), (5.0, "low"), (20.0, "slightly_low"), (50.0, "typical"),
    (80.0, "slightly_high"), (95.0, "high"), (99.0, "exceptional_high"),
])
def test_classification_bands(p, key):
    assert nm.classify(p)[0] == key


def test_classification_matches_extreme_thresholds():
    assert nm.classify(nm.EXTREME_LOW)[0] != "exceptional_low"
    assert nm.classify(nm.EXTREME_HIGH)[0] != "exceptional_high"
    assert nm.classify(nm.EXTREME_HIGH + 0.1)[0] == "exceptional_high"


def test_window_is_circular():
    assert nm._window(12, 0, 12, 1) == [12]
    assert nm._window(12, 1, 12, 1) == [1, 11, 12]
    assert nm._window(0, 1, 24, 0) == [0, 1, 23]
    assert nm._window(5, 6, 12, 1) == list(range(1, 13))        # todo el año


# ---------------------------------------------------------------- con BD
@pytest.mark.db
def test_percentile_matches_manual_comparable_window(rich_city):
    slug, _, past = rich_city
    observed_at = utc(2020, 6, 15, 12)
    value = 24.0
    with SessionLocal() as s:
        out = nm.analyze_normality(
            s, slug, observed_at=observed_at,
            values={"temperature_2m": value, "relative_humidity_2m": 55.0,
                    "wind_speed_10m": 10.0, "precipitation": 0.0, "surface_pressure": 1012.0},
        )
    assert out["status"] in ("ok", "partial")
    sub = past[(past["observed_at"].dt.month == 6) & past["observed_at"].dt.hour.isin([11, 12, 13])]
    expected = 100.0 * float((sub["temperature_2m"] <= value).mean())
    temp = next(v for v in out["variables"] if v["variable"] == "temperature_2m")
    assert temp["percentile"] == pytest.approx(expected, abs=0.1)
    assert out["reference"]["n_observations"] == len(sub)
    assert out["reference"]["widened"] is False
    assert out["reference"]["representativeness"] == "alta"


@pytest.mark.db
def test_reference_never_uses_future_rows(rich_city):
    """Las filas de 2021 (40 °C) existen en la BD pero son posteriores al instante analizado."""
    slug, _, _ = rich_city
    observed_at = utc(2020, 6, 15, 12)
    with SessionLocal() as s:
        out = nm.analyze_normality(
            s, slug, observed_at=observed_at,
            values={"temperature_2m": 24.0, "relative_humidity_2m": 55.0,
                    "wind_speed_10m": 10.0, "precipitation": 0.0, "surface_pressure": 1012.0},
        )
    assert out["reference"]["period_end"] < observed_at
    temp = next(v for v in out["variables"] if v["variable"] == "temperature_2m")
    assert temp["median"] < 35          # con fuga la mediana se dispararía hacia los 40 °C


@pytest.mark.db
def test_as_of_changes_reference_in_the_past(rich_city):
    """Reproducible: con as_of antiguo la referencia solo contiene datos anteriores a esa fecha."""
    slug, _, _ = rich_city
    vals = {"temperature_2m": 20.0, "relative_humidity_2m": 55.0, "wind_speed_10m": 10.0,
            "precipitation": 0.0, "surface_pressure": 1012.0}
    with SessionLocal() as s:
        old = nm.analyze_normality(s, slug, observed_at=utc(2020, 6, 15, 12), values=vals,
                                   as_of=utc(2017, 1, 1))
        again = nm.analyze_normality(s, slug, observed_at=utc(2020, 6, 15, 12), values=vals,
                                     as_of=utc(2017, 1, 1))
    assert old["reference"]["period_end"] < utc(2017, 1, 1)
    assert old["reference"]["n_observations"] == again["reference"]["n_observations"]
    assert old["variables"][0]["percentile"] == again["variables"][0]["percentile"]


@pytest.mark.db
def test_window_widens_with_flag_when_sample_is_small(city_factory):
    df = make_city_rows("2016-01-01", "2018-01-01", hours=(11, 12, 13))
    slug, _ = city_factory(df)
    nm.clear_cache()
    with SessionLocal() as s:
        out = nm.analyze_normality(
            s, slug, observed_at=utc(2018, 6, 15, 12),
            values={"temperature_2m": 18.0, "relative_humidity_2m": 55.0,
                    "wind_speed_10m": 10.0, "precipitation": 0.0, "surface_pressure": 1012.0},
        )
    ref = out["reference"]
    assert ref["widened"] is True and ref["window_step"] >= 1
    assert ref["n_observations"] >= nm.MIN_SAMPLE
    assert ref["representativeness"] in ("alta", "media", "baja")


@pytest.mark.db
def test_insufficient_history_returns_unavailable_not_a_conclusion(city_factory):
    df = make_city_rows("2018-06-01", "2018-06-08")          # una semana
    slug, _ = city_factory(df)
    nm.clear_cache()
    with SessionLocal() as s:
        out = nm.analyze_normality(
            s, slug, observed_at=utc(2018, 6, 9, 12),
            values={"temperature_2m": 18.0, "relative_humidity_2m": 55.0,
                    "wind_speed_10m": 10.0, "precipitation": 0.0, "surface_pressure": 1012.0},
        )
    assert out["status"] == "insufficient"
    assert out["variables"] == []
    assert "no se emite" in out["summary"].lower()


@pytest.mark.db
def test_city_without_observations_is_unavailable(city_factory):
    slug, _ = city_factory(None)
    with SessionLocal() as s:
        out = nm.analyze_normality(s, slug)
    assert out["status"] == "unavailable"


@pytest.mark.db
def test_dry_hour_has_no_invented_precipitation_percentile(rich_city):
    slug, _, _ = rich_city
    with SessionLocal() as s:
        out = nm.analyze_normality(
            s, slug, observed_at=utc(2020, 6, 15, 12),
            values={"temperature_2m": 20.0, "relative_humidity_2m": 55.0,
                    "wind_speed_10m": 10.0, "precipitation": 0.0, "surface_pressure": 1012.0},
        )
    rain = next(v for v in out["variables"] if v["variable"] == "precipitation")
    assert rain["classification"] == "dry"
    assert rain.get("percentile") is None
    assert not rain.get("is_extreme", False)


@pytest.mark.db
def test_cities_are_isolated(rich_city, city_factory):
    """Otra ciudad con temperaturas muy distintas no altera la referencia de la primera."""
    slug_a, _, past = rich_city
    hot = make_city_rows("2015-01-01", "2020-01-01", base=35.0, seed=3)
    slug_b, _ = city_factory(hot)
    nm.clear_cache()
    vals = {"temperature_2m": 24.0, "relative_humidity_2m": 55.0, "wind_speed_10m": 10.0,
            "precipitation": 0.0, "surface_pressure": 1012.0}
    with SessionLocal() as s:
        a = nm.analyze_normality(s, slug_a, observed_at=utc(2020, 6, 15, 12), values=vals)
        b = nm.analyze_normality(s, slug_b, observed_at=utc(2020, 6, 15, 12), values=vals)
    pa = next(v for v in a["variables"] if v["variable"] == "temperature_2m")["percentile"]
    pb = next(v for v in b["variables"] if v["variable"] == "temperature_2m")["percentile"]
    assert pa > pb + 20         # 24 °C es normal-alto en A y muy bajo en la ciudad cálida B
    sub = past[(past["observed_at"].dt.month == 6) & past["observed_at"].dt.hour.isin([11, 12, 13])]
    assert a["reference"]["n_observations"] == len(sub)


@pytest.mark.db
def test_latest_observation_ignores_prefilled_future_hours(city_factory):
    """El backfill guarda la hora siguiente con el valor previsto: no es «el tiempo actual»."""
    from datetime import UTC, datetime, timedelta

    from backend.app.db.models import WeatherObservation
    from backend.app.db.queries import latest_observation

    _, loc_id = city_factory(None)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    with SessionLocal() as s:
        for hours, temp in ((-1, 20.0), (0, 21.0), (1, 99.0)):
            s.add(WeatherObservation(location_id=loc_id, observed_at=now + timedelta(hours=hours),
                                     source="realtime", temperature_2m=temp))
        s.flush()
        assert float(latest_observation(s, loc_id).temperature_2m) == 21.0
        s.rollback()
