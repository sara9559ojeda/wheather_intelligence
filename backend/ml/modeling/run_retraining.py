"""CLI: comprueba si toca reentrenar y, si procede (o con --force), lo ejecuta.

Uso:
    python -m backend.ml.modeling.run_retraining              # solo si el disparador salta
    python -m backend.ml.modeling.run_retraining --force      # fuerza un reentrenamiento ahora
    python -m backend.ml.modeling.run_retraining --check-only # solo informa, no reentrena
"""

from __future__ import annotations

import argparse
import logging

from backend.app.db.session import session_scope
from backend.ml.config_loader import load_config
from backend.ml.modeling.retrain import check_trigger, run_retraining

log = logging.getLogger("modeling.run_retraining")
_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    p = argparse.ArgumentParser(description="Reentrenamiento con criterio de promoción (fase 9).")
    p.add_argument("--force", action="store_true",
                   help="Reentrena aunque no toque por disparador.")
    p.add_argument("--check-only", action="store_true",
                   help="Solo evalúa el disparador; no reentrena.")
    args = p.parse_args()

    cfg = load_config()
    with session_scope() as session:
        trigger = check_trigger(session, cfg.slug)
        log.info("Disparador: should_run=%s reason=%s n_new=%d desde=%s",
                 trigger.should_run, trigger.reason,
                 trigger.n_new_observations, trigger.previous_cutoff)

        if args.check_only:
            return
        if not trigger.should_run and not args.force:
            log.info("No toca reentrenar todavía (usa --force para forzarlo).")
            return

        reason = "manual" if args.force and not trigger.should_run else trigger.reason
        run = run_retraining(session, cfg.slug, reason=reason)

    log.info("Reentrenamiento %s | id=%d | duración=%.1fs",
             run.status, run.id, run.duration_seconds or 0)
    for horizon, r in (run.results or {}).items():
        mark = "✅ PROMOCIONADO" if r["promoted"] else "— conservado"
        log.info("  H=%3sh  %-24s rmse=%.3f  %s  (%s)",
                 horizon, r["challenger_type"], r["challenger_test_rmse"], mark, r["reason"])


if __name__ == "__main__":
    main()
