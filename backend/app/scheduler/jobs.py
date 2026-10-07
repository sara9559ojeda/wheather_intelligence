"""Trabajos periódicos (APScheduler) — ingesta en tiempo real."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from backend.app.core.config import settings
from backend.app.db.session import SessionLocal
from backend.app.services.events import bus
from backend.app.services.ingestion.open_meteo import OpenMeteoForecastClient
from backend.app.services.ingestion.realtime import ingest_current

log = logging.getLogger("scheduler")

_scheduler: BackgroundScheduler | None = None


def _make_client() -> OpenMeteoForecastClient:
    return OpenMeteoForecastClient(
        base_url=settings.open_meteo_forecast_url,
        latitude=settings.wi_latitude,
        longitude=settings.wi_longitude,
    )


def ingest_job() -> None:
    session = SessionLocal()
    try:
        summary = ingest_current(session, _make_client(), settings.wi_location_slug)
        session.commit()
        if summary:
            comp = summary["comparison"]
            log.info("Nueva observación %s | anomalía=%s | cluster=%s | %d predicciones",
                     summary["observed_at"], comp.get("is_anomaly"),
                     comp.get("cluster_label"), len(summary["predictions"]))
        else:
            log.debug("Sin datos nuevos de Open-Meteo")
    except Exception:  # noqa: BLE001 — el job nunca debe tumbar el scheduler
        session.rollback()
        log.exception("Fallo en el job de ingesta")
    finally:
        session.close()


def retrain_check_job() -> None:
    """Comprueba el disparador de reentrenamiento (fase 9) y reentrena si toca.

    El *check* es barato (una consulta); el reentrenamiento en sí (varios
    minutos) solo ocurre cuando de verdad hay volumen o calendario suficiente.
    """
    from backend.ml.modeling.retrain import check_trigger, run_retraining

    session = SessionLocal()
    try:
        trigger = check_trigger(session, settings.wi_location_slug)
        if not trigger.should_run:
            log.debug("Reentrenamiento: no toca (%d obs. nuevas desde %s)",
                      trigger.n_new_observations, trigger.previous_cutoff)
            return

        log.info("Reentrenamiento disparado (%s, %d obs. nuevas) — puede tardar varios minutos…",
                  trigger.reason, trigger.n_new_observations)
        run = run_retraining(session, settings.wi_location_slug, reason=trigger.reason)
        session.commit()
        promoted = [h for h, r in (run.results or {}).items() if r["promoted"]]
        log.info("Reentrenamiento %s | horizontes promocionados: %s",
                 run.status, promoted or "ninguno")
        if promoted:
            bus.publish("model_updated", {"status": run.status, "promoted_horizons": promoted})
    except Exception:  # noqa: BLE001 — igual que ingest_job, nunca tumba el scheduler
        session.rollback()
        log.exception("Fallo en el job de reentrenamiento")
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
