"""Reentrenamiento con criterio de promoción (fase 9, fase-1 §17).

Nada de MLOps pesado: un *script* que se puede llamar a mano o desde el
scheduler, y una regla de promoción **escrita y determinista**.

Procedimiento:

    1. consolidar datos = histórico + tiempo real acumulado hasta ahora
    2. validar calidad (duplicados, valores fuera de rango) -> abortar si falla
    3. re-cortar la partición cronológica con las MISMAS proporciones (70/15/15)
    4. entrenar un "retador" por horizonte (mismo procedimiento que la fase 5)
    5. evaluar al "campeón" activo sobre el MISMO test nuevo (comparación justa)
    6. promover el retador SOLO si mejora el RMSE >= epsilon, o si el campeón
       activo ha degradado más allá de un umbral y el retador no es peor
    7. registrar el intento (promovido o no) en `retraining_runs`, con el
       artefacto anterior archivado (nunca borrado)
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import joblib
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.bulk import bulk_upsert
from backend.app.db.models import ModelRun, Prediction, RetrainingRun, WeatherObservation
from backend.app.db.queries import get_location, load_observations_df
from backend.app.services.features.builder import (
    HORIZONS_HOURS,
    add_targets,
    build_feature_frame,
    list_feature_columns,
    target_column,
)
from backend.ml.config_loader import ARTIFACTS_DIR, PROJECT_ROOT
from backend.ml.modeling.baselines import SeasonalBaseline
from backend.ml.modeling.metrics import all_metrics
from backend.ml.modeling.train import HorizonResult, model_run_payload, train_horizon
from backend.ml.preparation.clean import clean_observations
from backend.ml.preparation.split import SplitResult, chronological_split

log = logging.getLogger(__name__)

QUALITY_MAX_CLIP_RATE = 0.05   # más de un 5% de valores recortados -> no fiable
QUALITY_MAX_DUP_RATE = 0.01    # más de un 1% de timestamps duplicados -> no fiable
_ACTIVE_MODEL_TYPES_EXCLUDE = "baseline%"


# --------------------------------------------------------------------------- #
#  Disparador
# --------------------------------------------------------------------------- #
@dataclass
class TriggerCheck:
    should_run: bool
    reason: str
    n_new_observations: int
    previous_cutoff: datetime | None


def _previous_cutoff(session: Session, location_id: int) -> tuple[datetime | None, datetime | None]:
    """(cutoff de la última ejecución, cuándo se hizo). None si nunca se ha reentrenado."""
    last = session.scalar(
        select(RetrainingRun)
        .where(RetrainingRun.location_id == location_id)
        .order_by(RetrainingRun.id.desc())
        .limit(1)
    )
    if last is not None and last.data_cutoff is not None:
        return last.data_cutoff, last.triggered_at
    # nunca se ha reentrenado: el "cutoff" es el final del histórico original
    cutoff = session.scalar(
        select(func.max(WeatherObservation.observed_at)).where(
            WeatherObservation.location_id == location_id,
            WeatherObservation.source == "historical",
        )
    )
    return cutoff, None


def _count_new_observations(session: Session, location_id: int, since: datetime | None) -> int:
    stmt = select(func.count()).select_from(WeatherObservation).where(
        WeatherObservation.location_id == location_id
    )
    if since is not None:
        stmt = stmt.where(WeatherObservation.observed_at > since)
    return int(session.scalar(stmt) or 0)


def check_trigger(session: Session, slug: str) -> TriggerCheck:
    """Decide si toca reentrenar: por volumen de datos nuevos o por calendario."""
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    cutoff, last_run_at = _previous_cutoff(session, loc.id)
    n_new = _count_new_observations(session, loc.id, cutoff)

    if n_new >= settings.retrain_volume_trigger:
        return TriggerCheck(True, "volume", n_new, cutoff)

    calendar_due = (
        last_run_at is None
        or (datetime.now(UTC) - last_run_at) >= timedelta(days=settings.retrain_calendar_days)
    )
    if calendar_due:
        return TriggerCheck(True, "calendar", n_new, cutoff)

    return TriggerCheck(False, "none", n_new, cutoff)


# --------------------------------------------------------------------------- #
#  Regla de promoción
# --------------------------------------------------------------------------- #
@dataclass
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    challenger_rmse: float,
    champion_rmse_on_new_test: float | None,
    champion_original_rmse: float | None,
    *,
    epsilon: float | None = None,
    degradation_threshold: float | None = None,
) -> PromotionDecision:
    """Regla escrita de promoción (fase-1 §17). Determinista, sin excepciones ocultas."""
    epsilon = settings.retrain_epsilon if epsilon is None else epsilon
    if degradation_threshold is None:
        degradation_threshold = settings.retrain_degradation_threshold
    if champion_rmse_on_new_test is None:
        return PromotionDecision(
            True, "no había un campeón activo evaluable para este horizonte; se promueve el retador"
        )

    required = champion_rmse_on_new_test * (1 - epsilon)
    if challenger_rmse <= required:
        gain = 100 * (1 - challenger_rmse / champion_rmse_on_new_test)
        return PromotionDecision(
            True,
            f"el retador mejora el RMSE un {gain:.1f}% "
            f"({challenger_rmse:.3f} vs {champion_rmse_on_new_test:.3f}; "
            f"se exigía >= {epsilon * 100:.0f}%)",
        )

    if champion_original_rmse is not None:
        degraded = champion_rmse_on_new_test > champion_original_rmse * degradation_threshold
        if degraded and challenger_rmse <= champion_rmse_on_new_test:
            return PromotionDecision(
                True,
                f"el campeón activo ha degradado (RMSE {champion_rmse_on_new_test:.3f} en el test "
                f"nuevo vs {champion_original_rmse:.3f} original, umbral "
                f"×{degradation_threshold:.2f}) y el retador no es peor",
            )

    return PromotionDecision(
        False,
        f"el retador no mejora lo suficiente (RMSE {challenger_rmse:.3f} vs "
        f"{champion_rmse_on_new_test:.3f} del campeón; se exige mejora "
        f">= {epsilon * 100:.0f}%)",
    )


# --------------------------------------------------------------------------- #
#  Evaluación del campeón activo sobre el test nuevo
# --------------------------------------------------------------------------- #
def _score_active_champion(
    session: Session, horizon: int, test_df: pd.DataFrame
) -> tuple[ModelRun | None, float | None, float | None]:
    active = session.scalar(
        select(ModelRun).where(
            ModelRun.horizon_hours == horizon,
            ModelRun.is_active.is_(True),
            ModelRun.model_type.notlike(_ACTIVE_MODEL_TYPES_EXCLUDE),
        )
    )
    if active is None or not active.artifact_path:
        return active, None, None

    artifact_file = PROJECT_ROOT / active.artifact_path
    if not artifact_file.exists():
        log.warning("H=%dh: artefacto activo %s no existe en disco", horizon, artifact_file)
        return active, None, None

    old = joblib.load(artifact_file)
    tcol = target_column(horizon)
    d = test_df.dropna(subset=[tcol])
    X = d[old["feature_columns"]]
    y = d[tcol].to_numpy(dtype="float64")
    rmse_new_test = all_metrics(y, old["pipeline"].predict(X))["rmse"]

    original_rmse = None
    if active.metrics:
        original_rmse = (active.metrics.get("test") or {}).get("rmse")

    return active, rmse_new_test, original_rmse


# --------------------------------------------------------------------------- #
#  Promoción efectiva: archivar artefacto viejo, guardar el nuevo, activar fila
# --------------------------------------------------------------------------- #
def _promote(
    session: Session,
    location_id: int,
    res: HorizonResult,
    feature_cols: list[str],
    split: SplitResult,
    periods: dict[str, str],
    previous_champion: ModelRun | None,
) -> ModelRun:
    if previous_champion is not None:
        previous_champion.is_active = False

    current_path = ARTIFACTS_DIR / f"model_temp_h{res.horizon}.joblib"
    if current_path.exists():
        archive_dir = ARTIFACTS_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        archived_name = f"model_temp_h{res.horizon}_{stamp}.joblib"
        shutil.move(str(current_path), str(archive_dir / archived_name))

    joblib.dump(
        {
            "pipeline": res.champion, "model_type": res.champion_type,
            "horizon_hours": res.horizon, "feature_columns": feature_cols,
            "residual_p05": res.residual_p05, "residual_p95": res.residual_p95,
            "trained_at": datetime.now(UTC).isoformat(),
        },
        current_path,
    )

    payload = model_run_payload(res, res.champion_type, is_champion=True)
    new_run = ModelRun(
        **{k: payload[k] for k in ("model_type", "horizon_hours", "target",
                                   "mae", "rmse", "r2", "hyperparams", "metrics", "is_active")},
        train_period=periods["train"], valid_period=periods["valid"], test_period=periods["test"],
        feature_list=feature_cols,
        artifact_path=f"backend/ml/artifacts/model_temp_h{res.horizon}.joblib",
    )
    session.add(new_run)
    session.flush()

    tcol = target_column(res.horizon)
    te = split.test.dropna(subset=[tcol])
    pred = res.champion.predict(te[feature_cols])
    base = pd.to_datetime(te["timestamp"], utc=True)
    records = [
        {
            "location_id": location_id, "model_run_id": new_run.id,
            "base_time": bt.to_pydatetime(),
            "target_time": (bt + timedelta(hours=res.horizon)).to_pydatetime(),
            "predicted_temperature": round(float(p), 2),
            "lower_bound": round(float(p + res.residual_p05), 2),
            "upper_bound": round(float(p + res.residual_p95), 2),
            "actual_temperature": round(float(a), 2),
        }
        for bt, p, a in zip(base, pred, te[tcol].to_numpy(), strict=True)
    ]
    bulk_upsert(session, Prediction, records,
                conflict_index=["model_run_id", "base_time"],
                label=f"predictions H={res.horizon}h (retador promovido)")
    return new_run


def _record_challenger(
    session: Session, res: HorizonResult, feature_cols: list[str], periods: dict[str, str]
) -> None:
    """Deja constancia del intento aunque no se promocione (trazabilidad, fase-1 §17)."""
    payload = model_run_payload(res, res.champion_type, is_champion=False)
    payload["metrics"]["test_if_promoted"] = res.test_champion  # se guarda igualmente, sin activar
    session.add(ModelRun(
        **{k: payload[k] for k in ("model_type", "horizon_hours", "target",
                                   "mae", "rmse", "r2", "hyperparams", "metrics", "is_active")},
        train_period=periods["train"], valid_period=periods["valid"], test_period=periods["test"],
        feature_list=feature_cols, artifact_path=None,
    ))


def _periods(split: SplitResult) -> dict[str, str]:
    b = split.boundaries
    return {n: f"{b[n]['start'][:10]}…{b[n]['end'][:10]}" for n in ("train", "valid", "test")}


# --------------------------------------------------------------------------- #
#  Orquestador
# --------------------------------------------------------------------------- #
def run_retraining(session: Session, slug: str, *, reason: str = "manual") -> RetrainingRun:
    t0 = time.time()
    loc = get_location(session, slug)
    if loc is None:
        raise ValueError(f"Ubicación '{slug}' no encontrada")

    previous_cutoff, _ = _previous_cutoff(session, loc.id)
    n_new = _count_new_observations(session, loc.id, previous_cutoff)

    raw = load_observations_df(session, slug)  # histórico + tiempo real, todo
    data_cutoff = raw["timestamp"].max().to_pydatetime()

    clean_df, creport = clean_observations(raw)
    clip_total = sum(creport.clipped.values())
    n_raw = max(len(raw), 1)
    clip_rate = clip_total / n_raw
    dup_rate = creport.duplicates_removed / n_raw
    validation_report = {
        "n_rows_raw": len(raw), "n_rows_clean": len(clean_df),
        "duplicates_removed": creport.duplicates_removed,
        "duplicate_rate": round(dup_rate, 4),
        "clipped": {k: v for k, v in creport.clipped.items() if v},
        "clip_rate": round(clip_rate, 4),
    }
    log.info("Reentrenamiento [%s]: %d filas nuevas desde %s | %s",
             reason, n_new, previous_cutoff, validation_report)

    if clip_rate > QUALITY_MAX_CLIP_RATE or dup_rate > QUALITY_MAX_DUP_RATE:
        run = RetrainingRun(
            location_id=loc.id, trigger_reason=reason, n_new_observations=n_new,
            data_cutoff=data_cutoff, status="aborted_validation",
            validation_report=validation_report, results=None,
            duration_seconds=round(time.time() - t0, 2),
        )
        session.add(run)
        session.flush()
        log.warning("Reentrenamiento ABORTADO por calidad de datos: %s", validation_report)
        return run

    feats = build_feature_frame(clean_df)
    feats = add_targets(feats, HORIZONS_HOURS)
    feature_cols = list_feature_columns(feats)
    split = chronological_split(feats)
    periods = _periods(split)
    seasonal = SeasonalBaseline().fit(split.train)

    results: dict[str, dict] = {}
    any_promoted = False
    for h in HORIZONS_HOURS:
        res = train_horizon(split.train, split.valid, split.test, feature_cols, h, seasonal)
        active, champion_rmse_new, champion_original_rmse = _score_active_champion(
            session, h, split.test
        )
        decision = decide_promotion(
            res.test_champion["rmse"], champion_rmse_new, champion_original_rmse
        )
        results[str(h)] = {
            "challenger_type": res.champion_type,
            "challenger_test_rmse": res.test_champion["rmse"],
            "champion_rmse_on_new_test": champion_rmse_new,
            "champion_original_test_rmse": champion_original_rmse,
            "promoted": decision.promote,
            "reason": decision.reason,
        }
        log.info("H=%2dh  retador=%s (rmse=%.3f)  campeón_en_test_nuevo=%s  -> %s",
                 h, res.champion_type, res.test_champion["rmse"],
                 f"{champion_rmse_new:.3f}" if champion_rmse_new is not None else "n/d",
                 "PROMOCIONADO" if decision.promote else "conservado")

        if decision.promote:
            any_promoted = True
            _promote(session, loc.id, res, feature_cols, split, periods, active)
        else:
            _record_challenger(session, res, feature_cols, periods)

    run = RetrainingRun(
        location_id=loc.id, trigger_reason=reason, n_new_observations=n_new,
        data_cutoff=data_cutoff, status=("promoted" if any_promoted else "not_promoted"),
        validation_report=validation_report, results=results,
        duration_seconds=round(time.time() - t0, 2),
    )
    session.add(run)
    session.flush()
    return run


__all__ = [
    "PromotionDecision",
    "TriggerCheck",
    "check_trigger",
    "decide_promotion",
    "run_retraining",
]
