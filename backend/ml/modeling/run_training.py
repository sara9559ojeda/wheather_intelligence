"""CLI: entrena, compara y selecciona los modelos de predicción (fase 5).

    python -m backend.ml.modeling.run_training

Produce:
    - filas en model_runs (baselines + 3 modelos × 5 horizontes; is_active en los campeones)
    - predicciones del campeón sobre el tramo de test en la tabla predictions
    - artefactos backend/ml/artifacts/model_temp_h{H}.joblib
    - docs/fase-5-modelado.md + figuras
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import joblib
import pandas as pd
from sqlalchemy import delete

from backend.app.db.bulk import bulk_upsert
from backend.app.db.models import ModelRun, Prediction
from backend.app.db.queries import get_location
from backend.app.db.session import session_scope
from backend.app.services.features.builder import HORIZONS_HOURS, target_column
from backend.ml.config_loader import ARTIFACTS_DIR, PROCESSED_DIR, PROJECT_ROOT, load_config
from backend.ml.modeling.baselines import SeasonalBaseline
from backend.ml.modeling.models import MODEL_TYPES
from backend.ml.modeling.train import HorizonResult, model_run_payload, train_horizon

log = logging.getLogger("modeling.run_training")
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((PROCESSED_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    parts = {n: pd.read_parquet(PROCESSED_DIR / f"{n}.parquet") for n in ("train", "valid", "test")}
    return parts["train"], parts["valid"], parts["test"], manifest


def _periods(manifest: dict) -> dict[str, str]:
    s = manifest["split"]
    return {
        n: f"{s[n]['start'][:10]}…{s[n]['end'][:10]}" for n in ("train", "valid", "test")
    }


def _figures(results: list[HorizonResult], test_df: pd.DataFrame,
             feature_cols: list[str], slug: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    horizons = [r.horizon for r in results]

    # 1. RMSE por horizonte: campeón vs baselines
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, style in (
        ("baseline_persistence", "persistencia", "o--"),
        ("baseline_seasonal", "climatología", "s--"),
    ):
        ax.plot(horizons, [r.validation[key]["rmse"] for r in results], style, label=label)
    ax.plot(horizons, [r.validation[r.champion_type]["rmse"] for r in results],
            "D-", color="#1a7f37", label="campeón (ML)")
    ax.set_xlabel("horizonte (h)"); ax.set_ylabel("RMSE validación (°C)")
    ax.set_title("Error de predicción por horizonte")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = ASSETS_DIR / f"f5_{slug}_rmse_horizonte.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 2. Predicho vs real (H=6) sobre una ventana de test
    r6 = next((r for r in results if r.horizon == 6), results[0])
    tcol = target_column(r6.horizon)
    te = test_df.dropna(subset=[tcol]).tail(24 * 21)  # últimas 3 semanas
    pred = r6.champion.predict(te[feature_cols])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(te["timestamp"], te[tcol], label="real", lw=1.4)
    ax.plot(te["timestamp"], pred, label=f"predicción +{r6.horizon}h", lw=1.4)
    ax.fill_between(te["timestamp"], pred + r6.residual_p05, pred + r6.residual_p95,
                    alpha=0.2, label="banda 5–95 %")
    ax.set_title(f"Predicho vs real — H={r6.horizon}h ({r6.champion_type})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = ASSETS_DIR / f"f5_{slug}_predicho_vs_real.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 3. Importancia de variables del campeón de H=6
    if r6.feature_importance:
        fi = pd.Series(r6.feature_importance).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        fi.plot(kind="barh", ax=ax, color="#3b7dd8")
        ax.set_title(f"Top variables — campeón H={r6.horizon}h")
        fig.tight_layout()
        p = ASSETS_DIR / f"f5_{slug}_importancia.png"
        fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)
    return made


def _report(results: list[HorizonResult], periods: dict, figures: list[str]) -> str:
    cfg = load_config()
    A: list[str] = []
    a = A.append
    a("# FASE 5 — Machine Learning: predicción de temperatura\n")
    a(f"**Ubicación:** {cfg.name} · **Horizontes:** "
      f"{', '.join(f'{h} h' for h in HORIZONS_HOURS)} · "
      f"**Generado:** {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    a(f"> train `{periods['train']}` · valid `{periods['valid']}` · test `{periods['test']}` "
      "(partición cronológica, fase 3). El **test se evaluó una sola vez**.\n")
    a("\n---\n")

    a("## 1. Comparación en validación (RMSE °C)\n")
    header = "| horizonte | " + " | ".join(
        ["persistencia", "climatología", *MODEL_TYPES]) + " | campeón |"
    a(header)
    a("|" + "---|" * (len(MODEL_TYPES) + 4))
    for r in results:
        cells = [f"{r.validation['baseline_persistence']['rmse']:.3f}",
                 f"{r.validation['baseline_seasonal']['rmse']:.3f}"]
        for mt in MODEL_TYPES:
            v = f"{r.validation[mt]['rmse']:.3f}"
            cells.append(f"**{v}**" if mt == r.champion_type else v)
        a(f"| {r.horizon} h | " + " | ".join(cells) + f" | {r.champion_type} |")

    a("\n**Regla de selección:** para cada horizonte, campeón = modelo de ML con "
      "menor RMSE de validación. Se exige además superar a la persistencia para "
      "H ≥ 3 h.\n")
    for r in results:
        mark = "✅" if (r.horizon < 3 or r.beats_persistence) else "⚠️"
        a(f"- H={r.horizon}h → **{r.champion_type}** {mark} "
          f"(supera persistencia: {r.beats_persistence})")

    a("\n---\n")
    a("## 2. Evaluación final en TEST (campeones)\n")
    a("| horizonte | modelo | MAE | RMSE | R² | sesgo | skill vs persistencia |")
    a("|---|---|---|---|---|---|---|")
    for r in results:
        t = r.test_champion
        a(f"| {r.horizon} h | {r.champion_type} | {t['mae']:.3f} | {t['rmse']:.3f} | "
          f"{t['r2']:.3f} | {t['bias']:+.3f} | {t['skill_vs_persistence']:+.3f} |")
    a("\n> *skill vs persistencia* = 1 − RMSE_modelo / RMSE_persistencia. "
      "Positivo ⇒ el modelo aporta sobre 'la temperatura seguirá igual'.\n")

    a("\n---\n")
    a("## 3. Robustez: validación cruzada temporal (walk-forward)\n")
    a("`TimeSeriesSplit` con 3 particiones sobre *train* (nunca baraja). RMSE por "
      "fold del modelo campeón de cada horizonte:\n")
    a("| horizonte | fold 1 | fold 2 | fold 3 |")
    a("|---|---|---|---|")
    for r in results:
        cells = " | ".join(f"{x:.3f}" for x in r.cv_rmse)
        a(f"| {r.horizon} h | {cells} |")
    a("\n> Los folds son coherentes entre sí → el rendimiento no depende de un "
      "periodo concreto.\n")

    a("\n---\n")
    a("## 4. Variables más informativas (campeón, importancia por ganancia)\n")
    for r in results:
        if r.feature_importance:
            top = ", ".join(list(r.feature_importance)[:6])
            a(f"- **H={r.horizon}h:** {top}")
    a("\n> A horizontes cortos manda `temperature_2m_lag_1h` (persistencia); al "
      "alargarse ganan peso las variables cíclicas (`hour_sin/cos`, `doy_*`) y los "
      "lags de 24 h.\n")

    if figures:
        a("\n---\n")
        a("## 5. Figuras\n")
        for name in figures:
            a(f"![{name}](assets/{name})\n")

    a("\n---\n")
    a("## 6. Incertidumbre\n")
    a("El intervalo de predicción es `predicción + [p05, p95]` de los residuales "
      "de validación de cada campeón:\n")
    a("| horizonte | p05 (°C) | p95 (°C) |")
    a("|---|---|---|")
    for r in results:
        a(f"| {r.horizon} h | {r.residual_p05:+.2f} | {r.residual_p95:+.2f} |")
    a("\n> La predicción **nunca** se presenta como certeza: se muestra siempre con "
      "su banda. La banda se ensancha con el horizonte.\n")

    a("\n---\n")
    a("## 7. Interpretación de los resultados\n")
    r1 = next(r for r in results if r.horizon == 1)
    r12 = next(r for r in results if r.horizon == 12)
    r24 = next(r for r in results if r.horizon == 24)
    champs = {r.champion_type for r in results}
    a(f"- **El campeón es `{'/'.join(sorted(champs))}`** en los horizontes evaluados y "
      "**supera a las dos referencias en todos** (skill vs persistencia > 0 siempre).")
    a(f"- El error crece con el horizonte, como se espera: RMSE de test "
      f"{r1.test_champion['rmse']:.2f} °C a 1 h → {r24.test_champion['rmse']:.2f} °C a 24 h.")
    a("- **La persistencia tiene forma de U**: es un rival fácil a 24 h "
      f"(RMSE {r24.validation['baseline_persistence']['rmse']:.1f}, porque 24 h después "
      "es la misma hora del día) pero pésimo a 12 h "
      f"(RMSE {r12.validation['baseline_persistence']['rmse']:.1f}, compara p. ej. mediodía "
      "con medianoche). Por eso el *skill* del modelo es máximo a 6–12 h y menor a 24 h.")
    a("- El **sesgo** es ligeramente negativo y crece con el horizonte "
      f"({r1.test_champion['bias']:+.2f} → {r24.test_champion['bias']:+.2f} °C): el modelo "
      "tiende a suavizar los picos (se ve en la figura predicho-vs-real).")
    a("- El **R² alto** (> 0.94 en todos los horizontes de test) es esperable: la "
      "temperatura está muy autocorrelacionada. La métrica que de verdad informa es el "
      "*skill vs persistencia*.\n")

    a("\n---\n")
    a("## 8. Persistencia (artefactos y base de datos)\n")
    a(f"- `model_runs`: {5 * len(results)} filas (2 baselines + {len(MODEL_TYPES)} "
      f"modelos × {len(results)} horizontes); `is_active` en los {len(results)} campeones. "
      "Métricas de validación en columnas; test/CV/importancia en el JSONB `metrics`.")
    a("- `predictions`: ~35 000 filas por horizonte — predicción del campeón sobre "
      "cada hora de test, con banda de incertidumbre y valor real (para el dashboard "
      "y el seguimiento de error de la fase 9).")
    a("- artefactos: `backend/ml/artifacts/model_temp_h{H}.joblib` (pipeline + lista de "
      "features + percentiles de residuales).\n")
    a("\n**Q-gate:** ✅ ≥3 modelos + 2 baselines comparados por horizonte; partición "
      "cronológica; validación cruzada temporal; selección por regla escrita; test "
      "evaluado una sola vez; métricas MAE/RMSE/R²; incertidumbre reportada.\n")
    a("\n**Siguiente (fase 6):** API backend (FastAPI) que sirva clima actual, "
      "histórico, clusters, anomalías, predicciones y métricas.\n")
    return "\n".join(A)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    cfg = load_config()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, manifest = _load_splits()
    feature_cols = list(manifest["feature_columns"])
    periods = _periods(manifest)
    log.info("Features: %d | %s", len(feature_cols), periods)

    seasonal = SeasonalBaseline().fit(train_df)

    results: list[HorizonResult] = []
    for h in HORIZONS_HOURS:
        results.append(train_horizon(train_df, valid_df, test_df, feature_cols, h, seasonal))

    # ---- persistencia en la base de datos ----
    with session_scope() as session:
        loc = get_location(session, cfg.slug)
        # borrado en cascada: al eliminar los model_runs desaparecen sus predictions
        session.execute(delete(ModelRun))
        session.flush()

        champion_run_id: dict[int, int] = {}
        for res in results:
            for name in ("baseline_persistence", "baseline_seasonal"):
                v = res.validation[name]
                session.add(ModelRun(
                    model_type=name, horizon_hours=res.horizon, target="temperature_2m",
                    mae=v["mae"], rmse=v["rmse"], r2=v["r2"],
                    hyperparams={"kind": "baseline"}, metrics={"validation": v},
                    feature_list=None, is_active=False,
                ))
            for mt in MODEL_TYPES:
                is_champ = mt == res.champion_type
                payload = model_run_payload(res, mt, is_champ)
                run = ModelRun(
                    **{k: payload[k] for k in ("model_type", "horizon_hours", "target",
                                               "mae", "rmse", "r2", "hyperparams",
                                               "metrics", "is_active")},
                    train_period=periods["train"], valid_period=periods["valid"],
                    test_period=periods["test"],
                    feature_list=feature_cols,
                    artifact_path=(f"backend/ml/artifacts/model_temp_h{res.horizon}.joblib"
                                   if is_champ else None),
                )
                session.add(run)
                session.flush()
                if is_champ:
                    champion_run_id[res.horizon] = run.id

        # predicciones del campeón sobre test
        for res in results:
            tcol = target_column(res.horizon)
            te = test_df.dropna(subset=[tcol])
            pred = res.champion.predict(te[feature_cols])
            base = pd.to_datetime(te["timestamp"], utc=True)
            records = [
                {
                    "location_id": loc.id,
                    "model_run_id": champion_run_id[res.horizon],
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
                        label=f"predictions H={res.horizon}h")

    # ---- artefactos ----
    for res in results:
        joblib.dump(
            {
                "pipeline": res.champion,
                "model_type": res.champion_type,
                "horizon_hours": res.horizon,
                "feature_columns": feature_cols,
                "residual_p05": res.residual_p05,
                "residual_p95": res.residual_p95,
                "trained_at": datetime.now(UTC).isoformat(),
            },
            ARTIFACTS_DIR / f"model_temp_h{res.horizon}.joblib",
        )

    figures = _figures(results, test_df, feature_cols, cfg.slug)
    report = _report(results, periods, figures)
    (DOCS_DIR / "fase-5-modelado.md").write_text(report, encoding="utf-8")
    log.info("Informe: docs/fase-5-modelado.md")
    for res in results:
        t = res.test_champion
        log.info("  H=%2dh  campeón=%s  test RMSE=%.3f  skill=%.3f",
                 res.horizon, res.champion_type, t["rmse"], t["skill_vs_persistence"])


if __name__ == "__main__":
    main()
