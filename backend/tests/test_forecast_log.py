"""F3/F4 — registro persistente de pronósticos, verificación sin fugas y calibración conformal."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from backend.app.db.models import (
    FORECAST_EXPIRED,
    FORECAST_PENDING,
    FORECAST_VERIFIED,
    ForecastLog,
    Location,
    WeatherObservation,
)
from backend.app.db.session import SessionLocal
from backend.app.services.evaluation import forecast_log as fl
from backend.app.services.evaluation import uncertainty as un
from backend.app.services.prediction.predictor import HorizonPrediction
from backend.tests.conftest import utc

BASE = utc(2026, 3, 1, 10)


def _pred(
    horizon: int = 3, base=BASE, value: float = 20.0, version: str = "v1"
) -> HorizonPrediction:
    b = pd.Timestamp(base)
    return HorizonPrediction(
        horizon_hours=horizon, base_time=b, target_time=b + pd.Timedelta(hours=horizon),
        predicted_temperature=value, lower_bound=value - 2, upper_bound=value + 2,
        model_type="ridge", model_version=version, interval_level=0.9,
        interval_method=un.METHOD_STATIC,
    )


def _obs(session, loc_id, when, temp, *, source="historical", ingested_at=None):
    row = WeatherObservation(location_id=loc_id, observed_at=when, source=source,
                             temperature_2m=temp)
    if ingested_at is not None:
        row.ingested_at = ingested_at
    session.add(row)
    session.flush()


def _count(session, loc_id, status=None) -> int:
    q = select(func.count()).select_from(ForecastLog).where(ForecastLog.location_id == loc_id)
    if status:
        q = q.where(ForecastLog.status == status)
    return int(session.scalar(q))


# ---------------------------------------------------------------- sin BD: estadística pura
def test_conformal_bounds_use_finite_sample_ranks():
    res = np.arange(1, 101, dtype=float)          # 1..100
    lo, hi = un.conformal_bounds(res, 0.90)
    assert lo == 5.0                              # floor(101 * 0.05) = 5
    assert hi == 96.0                             # ceil(101 * 0.95) = 96


def test_conformal_interval_reaches_nominal_coverage_in_time_order():
    """Calibra con el pasado y evalúa en el futuro (sin partición aleatoria)."""
    rng = np.random.default_rng(1)
    resid = rng.normal(0.3, 1.5, 6000)
    calib, test = resid[:1000], resid[1000:]
    lo, hi = un.conformal_bounds(calib, 0.90)
    coverage = float(np.mean((test >= lo) & (test <= hi)))
    assert 0.88 <= coverage <= 0.92


def test_conformal_requires_residuals():
    with pytest.raises(ValueError):
        un.conformal_bounds(np.array([]))


@pytest.mark.parametrize("kwargs,level", [
    (dict(n_live=200, live_coverage=0.91, skill=0.3, degraded=False, offline_coverage=0.9), "alta"),
    (dict(n_live=60, live_coverage=0.91, skill=0.3, degraded=False, offline_coverage=0.9), "media"),
    (dict(n_live=200, live_coverage=0.83, skill=0.3, degraded=False, offline_coverage=0.9),
     "media"),
    (dict(n_live=200, live_coverage=0.70, skill=0.3, degraded=False, offline_coverage=0.9), "baja"),
    # dos problemas a la vez (no supera a la persistencia + deterioro) bajan un nivel
    (dict(n_live=200, live_coverage=0.91, skill=-0.1, degraded=True, offline_coverage=0.9),
     "media"),
    (dict(n_live=200, live_coverage=0.83, skill=-0.1, degraded=False, offline_coverage=0.9),
     "baja"),
    (dict(n_live=5, live_coverage=0.8, skill=None, degraded=False, offline_coverage=0.9),
     "preliminar"),
    (dict(n_live=0, live_coverage=None, skill=None, degraded=False, offline_coverage=None),
     "no_evaluable"),
])
def test_reliability_label_is_based_on_verified_metrics(kwargs, level):
    assert un.classify_reliability(**kwargs)[0] == level


def test_reliability_never_high_with_few_cases():
    level, _, reasons = un.classify_reliability(
        n_live=fl.MIN_RELIABLE_N, live_coverage=0.90, skill=0.5, degraded=False,
        offline_coverage=0.9)
    assert level != "alta"
    assert reasons


# ---------------------------------------------------------------- BD: registro
@pytest.mark.db
def test_record_is_idempotent_and_persists_all_fields(city_factory):
    _, loc_id = city_factory(None)
    preds = [_pred(1), _pred(3), _pred(6)]
    with SessionLocal() as s:
        first = fl.record_forecasts(s, loc_id, preds, persistence_temp=19.0,
                                    reference={preds[1].target_time.to_pydatetime(): 21.5},
                                    champions={})
        again = fl.record_forecasts(s, loc_id, preds, persistence_temp=19.0, reference=None,
                                    champions={})
        s.commit()
        assert (first, again) == (3, 0)
        assert _count(s, loc_id) == 3                      # sin duplicados
        row = s.scalar(select(ForecastLog).where(ForecastLog.location_id == loc_id,
                                                 ForecastLog.horizon_hours == 3))
        assert row.status == FORECAST_PENDING and row.observed_temperature is None
        assert row.model_version == "v1" and row.model_type == "ridge"
        assert row.baseline_persistence == pytest.approx(19.0)
        assert row.reference_forecast == pytest.approx(21.5)
        assert row.lower_bound < row.predicted_temperature < row.upper_bound
        assert row.issued_at is not None


@pytest.mark.db
def test_nothing_is_verified_before_target_plus_grace(city_factory):
    _, loc_id = city_factory(None)
    p = _pred(3)
    with SessionLocal() as s:
        fl.record_forecasts(s, loc_id, [p], persistence_temp=None, reference=None, champions={})
        _obs(s, loc_id, p.target_time.to_pydatetime(), 18.0)
        # antes de que pase el margen, aunque ya exista una fila histórica
        too_early = p.target_time.to_pydatetime() + fl.GRACE - timedelta(minutes=1)
        assert fl.verify_pending(s, now=too_early) == {}
        assert _count(s, loc_id, FORECAST_PENDING) == 1
        ok = p.target_time.to_pydatetime() + fl.GRACE + timedelta(minutes=1)
        assert fl.verify_pending(s, now=ok) == {loc_id: 1}
        row = s.scalar(select(ForecastLog).where(ForecastLog.location_id == loc_id))
        assert row.status == FORECAST_VERIFIED
        assert row.observed_temperature == pytest.approx(18.0)
        assert row.error == pytest.approx(2.0)             # predicho − observado = 20 − 18
        assert row.observed_source == "historical"
        s.rollback()


@pytest.mark.db
def test_verification_is_idempotent(city_factory):
    _, loc_id = city_factory(None)
    p = _pred(1)
    with SessionLocal() as s:
        fl.record_forecasts(s, loc_id, [p], persistence_temp=None, reference=None, champions={})
        _obs(s, loc_id, p.target_time.to_pydatetime(), 19.0)
        now = p.target_time.to_pydatetime() + timedelta(hours=3)
        assert fl.verify_pending(s, now=now) == {loc_id: 1}
        first = s.scalar(select(ForecastLog.verified_at).where(ForecastLog.location_id == loc_id))
        assert fl.verify_pending(s, now=now + timedelta(hours=1)) == {}
        assert s.scalar(select(ForecastLog.verified_at)
                        .where(ForecastLog.location_id == loc_id)) == first
        s.rollback()


@pytest.mark.db
def test_realtime_row_stored_before_target_is_not_an_observation(city_factory):
    """El backfill guarda por adelantado el valor PREVISTO de horas futuras: no es observación."""
    _, loc_id = city_factory(None)
    p = _pred(3)
    target = p.target_time.to_pydatetime()
    with SessionLocal() as s:
        fl.record_forecasts(s, loc_id, [p], persistence_temp=None, reference=None, champions={})
        _obs(s, loc_id, target, 25.0, source="realtime", ingested_at=target - timedelta(hours=2))
        now = target + timedelta(hours=3)
        assert fl.verify_pending(s, now=now) == {}                       # sigue pendiente
        assert _count(s, loc_id, FORECAST_PENDING) == 1
        # la misma fila refrescada durante/tras la hora sí es una medición
        row = s.scalar(select(WeatherObservation).where(
            WeatherObservation.location_id == loc_id, WeatherObservation.source == "realtime"))
        row.ingested_at = target + timedelta(minutes=20)
        s.flush()
        assert fl.verify_pending(s, now=now) == {loc_id: 1}
        v = s.scalar(select(ForecastLog).where(ForecastLog.location_id == loc_id))
        assert v.observed_source == "realtime" and v.observed_temperature == pytest.approx(25.0)
        s.rollback()


@pytest.mark.db
def test_unobserved_forecast_expires_after_the_wait(city_factory):
    _, loc_id = city_factory(None)
    p = _pred(1, base=utc(2026, 1, 1, 0))
    with SessionLocal() as s:
        fl.record_forecasts(s, loc_id, [p], persistence_temp=None, reference=None, champions={})
        target = p.target_time.to_pydatetime()
        fl.verify_pending(s, now=target + fl.EXPIRE_AFTER - timedelta(days=1))
        assert _count(s, loc_id, FORECAST_PENDING) == 1
        fl.verify_pending(s, now=target + fl.EXPIRE_AFTER + timedelta(days=1))
        assert _count(s, loc_id, FORECAST_EXPIRED) == 1
        s.rollback()


@pytest.mark.db
def test_metrics_only_use_verified_forecasts(city_factory):
    slug, loc_id = city_factory(None)
    now = utc(2026, 3, 10, 12)
    with SessionLocal() as s:
        for i, (pred, obs) in enumerate([(20.0, 19.0), (20.0, 22.0), (20.0, 20.0)]):
            base = now - timedelta(hours=10 + i)
            p = _pred(1, base=base, value=pred)
            fl.record_forecasts(s, loc_id, [p], persistence_temp=obs - 1, reference=None,
                                champions={})
            _obs(s, loc_id, p.target_time.to_pydatetime(), obs)
        # dos pronósticos más, todavía sin observación → pendientes
        for i in range(2):
            fl.record_forecasts(s, loc_id, [_pred(1, base=now - timedelta(minutes=30 + i),
                                                  value=99.0)],
                                persistence_temp=None, reference=None, champions={})
        fl.verify_pending(s, now=now)
        df = fl.verified_frame(s, loc_id, 7, now=now)
        m = fl.horizon_metrics(df, 1, 7)
        assert m["n_verified"] == 3
        assert m["mae"] == pytest.approx((1 + 2 + 0) / 3, abs=1e-3)
        assert m["rmse"] == pytest.approx(np.sqrt((1 + 4 + 0) / 3), abs=1e-3)
        assert m["bias"] == pytest.approx((1 - 2 + 0) / 3, abs=1e-3)
        loc = s.get(Location, loc_id)
        summ = fl.evaluation_summary(s, loc, 7)
        assert summ["n_verified_total"] == 3 and summ["n_pending"] == 2
        s.rollback()


@pytest.mark.db
def test_summary_without_forecasts_is_honest_not_empty_metrics(city_factory):
    _, loc_id = city_factory(None)
    with SessionLocal() as s:
        loc = s.get(Location, loc_id)
        out = fl.evaluation_summary(s, loc, 30)
    assert out["status"] == "unavailable"
    assert out["n_verified_total"] == 0
    assert all(h["n_verified"] == 0 and "mae" not in h for h in out["horizons"])


@pytest.mark.db
def test_cities_do_not_share_forecast_logs(city_factory):
    _, a = city_factory(None)
    _, b = city_factory(None)
    with SessionLocal() as s:
        fl.record_forecasts(s, a, [_pred(1), _pred(3)], persistence_temp=None, reference=None,
                            champions={})
        fl.record_forecasts(s, b, [_pred(1)], persistence_temp=None, reference=None, champions={})
        s.commit()
        assert fl.evaluation_summary(s, s.get(Location, a), 30)["n_logged_total"] == 2
        assert fl.evaluation_summary(s, s.get(Location, b), 30)["n_logged_total"] == 1


# ---------------------------------------------------------------- BD: calibración
def _add_verified(session, loc_id, n: int, *, version: str, horizon: int = 3, seed: int = 0,
                  offset: int = 0):
    rng = np.random.default_rng(seed)
    start = utc(2026, 1, 1)
    for i in range(n):
        base = start + timedelta(hours=offset + i)
        err = float(rng.normal(0, 1.0))
        session.add(ForecastLog(
            location_id=loc_id, horizon_hours=horizon, base_time=base,
            target_time=base + timedelta(hours=horizon), predicted_temperature=20.0,
            lower_bound=18.0, upper_bound=22.0, interval_level=0.9,
            interval_method=un.METHOD_STATIC, model_type="ridge", model_version=version,
            status=FORECAST_VERIFIED, observed_temperature=20.0 - err, observed_source="historical",
            error=err,
        ))
    session.flush()


@pytest.mark.db
def test_calibration_switches_only_with_enough_same_version_residuals(city_factory):
    _, loc_id = city_factory(None)
    with SessionLocal() as s:
        _add_verified(s, loc_id, un.MIN_CALIBRATION - 1, version="v1")
        # otro modelo: no cuenta
        _add_verified(s, loc_id, 300, version="v0", seed=5, offset=1000)
        p = un.apply_calibration(s, loc_id, [_pred(3, version="v1")])[0]
        assert p.interval_method == un.METHOD_STATIC
        assert (p.lower_bound, p.upper_bound) == (18.0, 22.0)

        _add_verified(s, loc_id, 1, version="v1", seed=11, offset=5000)
        p = un.apply_calibration(s, loc_id, [_pred(3, version="v1")])[0]
        s.rollback()
    assert p.interval_method == un.METHOD_ROLLING
    assert p.interval_level == un.NOMINAL_LEVEL


@pytest.mark.db
def test_rolling_interval_is_centered_on_observed_residuals(city_factory):
    _, loc_id = city_factory(None)
    with SessionLocal() as s:
        _add_verified(s, loc_id, 400, version="v1", seed=3)
        p = un.apply_calibration(s, loc_id, [_pred(3, value=20.0, version="v1")])[0]
        s.rollback()
    assert p.interval_method == un.METHOD_ROLLING
    # residuales ~ N(0, 1): el intervalo del 90 % mide unas 3,3 desviaciones (no los ±2 estáticos)
    assert 2.6 < p.upper_bound - p.lower_bound < 4.0
    assert p.lower_bound < 20.0 < p.upper_bound


@pytest.mark.db
def test_pending_forecasts_never_feed_calibration(city_factory):
    _, loc_id = city_factory(None)
    with SessionLocal() as s:
        fl.record_forecasts(s, loc_id, [_pred(3, base=BASE + timedelta(hours=i))
                                        for i in range(150)],
                            persistence_temp=None, reference=None, champions={})
        res = un.recent_residuals(s, loc_id, 3, "v1")
        s.rollback()
    assert len(res) == 0
