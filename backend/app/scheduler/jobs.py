"""Trabajos periódicos (APScheduler) — ingesta en tiempo real.

En modo serverless (``ENABLE_SCHEDULER=false``) el scheduler no arranca y la
ingesta se dispara desde fuera con ``POST /api/tasks/ingest`` (ver ``run_ingest``).
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from backend.app.core.config import settings
from backend.app.db.session import SessionLocal
from backend.app.services.evaluation.forecast_log import verify_pending
from backend.app.services.events import bus
from backend.app.services.ingestion.open_meteo import OpenMeteoForecastClient
from backend.app.services.ingestion.realtime import ingest_current
from backend.app.services.locations import ready_locations

log = logging.getLogger("scheduler")

_scheduler: BackgroundScheduler | None = None


def _make_client(latitude: float, longitude: float) -> OpenMeteoForecastClient:
    return OpenMeteoForecastClient(
        base_url=settings.open_meteo_forecast_url,
        latitude=latitude,
        longitude=longitude,
    )


def active_locations() -> list[tuple[str, float, float]]:
    """(slug, latitud, longitud) de las ciudades con modelos; si no hay, la configurada."""
    session = SessionLocal()
    try:
        found = [(loc.slug, float(loc.latitude), float(loc.longitude))
                 for loc in ready_locations(session)]
    finally:
        session.close()
    return found or [(settings.wi_location_slug, settings.wi_latitude, settings.wi_longitude)]


def _ingest_location(slug: str, latitude: float, longitude: float) -> dict | None:
    """Una pasada de ingesta de UNA ciudad. Lanza la excepción si falla (tras rollback)."""
    session = SessionLocal()
    try:
        summary = ingest_current(session, _make_client(latitude, longitude), slug)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    if summary:
        comp = summary["comparison"]
        log.info("[%s] Nueva observación %s | anomalía=%s | cluster=%s | %d predicciones",
                 slug, summary["observed_at"], comp.get("is_anomaly"),
                 comp.get("cluster_label"), len(summary["predictions"]))
    else:
        log.debug("[%s] Sin datos nuevos de Open-Meteo", slug)
    return summary


def run_verification() -> dict[str, int]:
    """Verifica los pronósticos cuya hora objetivo ya pasó y avisa por el bus (idempotente)."""
    from sqlalchemy import select

    from backend.app.db.models import Location

    session = SessionLocal()
    try:
        by_loc = verify_pending(session)
        slugs = {}
        if by_loc:
            slugs = dict(session.execute(
                select(Location.id, Location.slug).where(Location.id.in_(list(by_loc)))).all())
        session.commit()
    except Exception:  # noqa: BLE001 — la verificación nunca debe tumbar la ingesta
        session.rollback()
        log.exception("Fallo verificando pronósticos")
        return {}
    finally:
        session.close()
    result = {slugs[i]: n for i, n in by_loc.items() if i in slugs}
    for slug, n in result.items():
        log.info("[%s] %d pronósticos verificados", slug, n)
        bus.publish("forecasts_verified", {"location": slug, "n_verified": n})
    return result


def run_ingest() -> dict | None:
    """Una pasada de ingesta de TODAS las ciudades activas (el fallo de una no frena a las demás).

    Devuelve el resumen de la ubicación predeterminada (o de la primera que haya ido bien).
    Lanza la excepción si fallan todas. La usan el job del scheduler y el endpoint
    ``POST /api/tasks/ingest`` (modo serverless).
    """
    targets = active_locations()
    summaries: dict[str, dict | None] = {}
    failures = 0
    for slug, lat, lon in targets:
        try:
            summaries[slug] = _ingest_location(slug, lat, lon)
        except Exception:  # noqa: BLE001 — una ciudad caída no debe frenar a las demás
            failures += 1
            log.exception("Fallo en la ingesta de %s", slug)
            if len(targets) == 1:
                raise
    run_verification()
    if failures == len(targets):
        raise RuntimeError("Falló la ingesta de todas las ubicaciones")
    return summaries.get(settings.wi_location_slug) or next(
        (s for s in summaries.values() if s), None
    )


def ingest_job() -> None:
    try:
        run_ingest()
    except Exception:  # noqa: BLE001 — el job nunca debe tumbar el scheduler
        log.exception("Fallo en el job de ingesta")


def retrain_check_job() -> None:
    """Comprueba el disparador de reentrenamiento (fase 9) y reentrena si toca.

    El *check* es barato (una consulta); el reentrenamiento en sí (varios
    minutos) solo ocurre cuando de verdad hay volumen o calendario suficiente.
    """
    from backend.ml.modeling.retrain import check_trigger, run_retraining

    for slug, _lat, _lon in active_locations():
        _retrain_if_due(slug, check_trigger, run_retraining)


def _retrain_if_due(slug: str, check_trigger, run_retraining) -> None:
    session = SessionLocal()
    try:
        trigger = check_trigger(session, slug)
        if not trigger.should_run:
            log.debug("[%s] Reentrenamiento: no toca (%d obs. nuevas desde %s)",
                      slug, trigger.n_new_observations, trigger.previous_cutoff)
            return

        log.info("[%s] Reentrenamiento disparado (%s, %d obs. nuevas) — puede tardar minutos…",
                 slug, trigger.reason, trigger.n_new_observations)
        run = run_retraining(session, slug, reason=trigger.reason)
        session.commit()
        promoted = [h for h, r in (run.results or {}).items() if r["promoted"]]
        log.info("[%s] Reentrenamiento %s | horizontes promocionados: %s",
                 slug, run.status, promoted or "ninguno")
        if promoted:
            bus.publish("model_updated", {
                "location": slug, "status": run.status, "promoted_horizons": promoted,
            })
    except Exception:  # noqa: BLE001 — igual que ingest_job, nunca tumba el scheduler
        session.rollback()
        log.exception("[%s] Fallo en el job de reentrenamiento", slug)
    finally:
        session.close()


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        ingest_job, "interval",
        minutes=settings.ingest_interval_minutes,
        id="realtime_ingest", max_instances=1, coalesce=True,
    )
    _scheduler.add_job(
        retrain_check_job, "interval",
        hours=settings.retrain_check_interval_hours,
        id="retrain_check", max_instances=1, coalesce=True,
    )
    _scheduler.start()
    log.info("Scheduler iniciado: ingesta cada %d min · comprobación de reentrenamiento cada %d h",
             settings.ingest_interval_minutes, settings.retrain_check_interval_hours)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
