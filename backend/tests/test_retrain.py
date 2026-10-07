"""Pruebas del reentrenamiento con criterio de promoción (fase 9).

La ejecución completa de ``run_retraining`` (5 horizontes × 3 modelos sobre
~234 000 filas) tarda varios minutos — no se ejecuta en la batería automática.
Se verificó manualmente con ``python -m backend.ml.modeling.run_retraining --force``
(ver docs/fase-9-retraining.md). Aquí se prueban, offline y a fondo, las
piezas de las que depende la corrección de la regla: la decisión de promoción
y el cálculo del disparador.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from backend.ml.modeling.retrain import decide_promotion


# --------------------------------------------------------------------------- #
#  decide_promotion — pura, sin BD
# --------------------------------------------------------------------------- #
def test_promotes_when_challenger_clearly_better():
    d = decide_promotion(1.0, 1.5, 1.5, epsilon=0.02)
    assert d.promote
    assert "mejora" in d.reason


def test_does_not_promote_when_improvement_below_epsilon():
    # 1.49 vs 1.50 es solo un 0.7% de mejora; se exige 2%
    d = decide_promotion(1.49, 1.50, 1.50, epsilon=0.02)
    assert not d.promote


def test_promotion_boundary_is_inclusive():
    # exactamente el 2% de mejora -> debe promocionar (<=, no <)
    champion = 1.00
    challenger = champion * (1 - 0.02)
    d = decide_promotion(challenger, champion, champion, epsilon=0.02)
    assert d.promote


def test_promotes_when_no_previous_champion_to_compare():
    d = decide_promotion(2.0, None, None)
    assert d.promote
    assert "no había un campeón" in d.reason


def test_promotes_when_champion_has_degraded_and_challenger_not_worse():
    # el campeón se entrenó con rmse 1.0 y ahora, en datos nuevos, tiene 1.30 (+30%,
    # por encima del umbral de degradación). El retador (1.29) no llega al 2% de
    # mejora exigido por la vía normal, pero tampoco es peor -> promociona por
    # la vía de degradación del campeón.
    d = decide_promotion(
        challenger_rmse=1.29, champion_rmse_on_new_test=1.30, champion_original_rmse=1.00,
        epsilon=0.02, degradation_threshold=1.20,
    )
    assert d.promote
    assert "degradado" in d.reason


def test_does_not_promote_if_champion_degraded_but_challenger_is_worse():
    d = decide_promotion(
        challenger_rmse=1.40, champion_rmse_on_new_test=1.30, champion_original_rmse=1.00,
        epsilon=0.02, degradation_threshold=1.20,
    )
    assert not d.promote


def test_does_not_promote_on_mild_degradation_within_threshold():
    # +10% de degradación, por debajo del umbral del 20% -> no dispara la vía de degradación
    d = decide_promotion(
        challenger_rmse=1.09, champion_rmse_on_new_test=1.10, champion_original_rmse=1.00,
        epsilon=0.02, degradation_threshold=1.20,
    )
    assert not d.promote


def test_uses_settings_defaults_when_epsilon_omitted():
    from backend.app.core.config import settings

    just_inside = 1.0 * (1 - settings.retrain_epsilon)
    just_outside = 1.0 * (1 - settings.retrain_epsilon / 2)
    assert decide_promotion(just_inside, 1.0, 1.0).promote
    assert not decide_promotion(just_outside, 1.0, 1.0).promote


# --------------------------------------------------------------------------- #
#  Disparador — necesita BD (solo lecturas, rápido)
# --------------------------------------------------------------------------- #
@pytest.mark.db
def test_check_trigger_reports_new_observation_count():
    """No asume si ya hubo un reentrenamiento antes (la tabla es real y evoluciona);
    solo comprueba que la respuesta es internamente consistente."""
    from backend.app.db.queries import get_location
    from backend.app.db.session import session_scope
    from backend.ml.modeling.retrain import check_trigger

    with session_scope() as s:
        loc = get_location(s, "madrid_barajas")
        assert loc is not None
        trigger = check_trigger(s, "madrid_barajas")
        assert trigger.n_new_observations >= 0
        assert trigger.reason in {"volume", "calendar", "none"}
        assert trigger.should_run == (trigger.reason in {"volume", "calendar"})
        if trigger.reason == "volume":
            from backend.app.core.config import settings

            assert trigger.n_new_observations >= settings.retrain_volume_trigger


@pytest.mark.db
def test_score_active_champion_matches_stored_test_metrics_order_of_magnitude():
    """El campeón activo, evaluado sobre SU PROPIO test, debe dar un RMSE similar
    al que ya tiene guardado en model_runs.metrics (no idéntico: aquí se recorta
    a las filas con target no nulo del parquet, no se reconstruye el split)."""
    from backend.app.db.session import session_scope
    from backend.ml.config_loader import PROCESSED_DIR
    from backend.ml.modeling.retrain import _score_active_champion

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    with session_scope() as s:
        active, rmse_new, original_rmse = _score_active_champion(s, 6, test_df)
        assert active is not None
        assert rmse_new is not None
        assert original_rmse is not None
        # deben ser del mismo orden de magnitud (mismo dataset de test, mismo modelo)
        assert abs(rmse_new - original_rmse) < 0.5 * original_rmse + 0.5


# --------------------------------------------------------------------------- #
#  Cutoff / conteo de observaciones nuevas
# --------------------------------------------------------------------------- #
@pytest.mark.db
def test_count_new_observations_since_a_future_cutoff_is_zero():
    from backend.app.db.queries import get_location
    from backend.app.db.session import session_scope
    from backend.ml.modeling.retrain import _count_new_observations

    with session_scope() as s:
        loc = get_location(s, "madrid_barajas")
        far_future = datetime.now(UTC) + timedelta(days=3650)
        assert _count_new_observations(s, loc.id, far_future) == 0
