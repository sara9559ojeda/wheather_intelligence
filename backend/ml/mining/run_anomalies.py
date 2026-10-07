"""CLI: detección de anomalías con Isolation Forest sobre residuales (fase 4).

    - estima la climatología (media por mes×hora) con el tramo de TRAIN
    - ajusta Isolation Forest sobre los residuales de train
    - puntúa TODAS las observaciones y persiste en la tabla `anomalies`
    - guarda el artefacto y genera docs/fase-4-anomalias.md + figuras

Uso:
    python -m backend.ml.mining.run_anomalies [--contamination 0.02]
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import joblib
import pandas as pd
from sqlalchemy import delete

from backend.app.db.bulk import bulk_upsert
from backend.app.db.models import Anomaly
from backend.app.db.queries import get_location, load_observations_df
from backend.app.db.session import session_scope
from backend.ml.config_loader import ARTIFACTS_DIR, PROCESSED_DIR, PROJECT_ROOT, load_config
from backend.ml.mining import anomalies as an

log = logging.getLogger("mining.anomalies")
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
DETECTOR = "isolation_forest"


def _figures(scored: pd.DataFrame, slug: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    s = scored.dropna(subset=["anomaly_score"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(s["anomaly_score"], bins=80, color="#3b7dd8")
    axes[0].axvline(s.loc[s["is_anomaly"], "anomaly_score"].max(), ls="--", color="red")
    axes[0].set_title("Distribución del anomaly_score")
    axes[0].set_xlabel("score (más negativo = más anómalo)")

    by_year = s.assign(year=s["timestamp"].dt.year).groupby("year")["is_anomaly"].mean() * 100
    axes[1].bar(by_year.index, by_year.values, color="#d8663b")
    axes[1].set_title("% de horas marcadas como anómalas, por año")
    axes[1].set_ylabel("%")
    fig.tight_layout()
    p = ASSETS_DIR / f"f4_{slug}_anomalias.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)
    return made


def _report(scored: pd.DataFrame, model: an.AnomalyModel, figures: list[str]) -> str:
    cfg = load_config()
    s = scored.dropna(subset=["anomaly_score"])
    n_anom = int(s["is_anomaly"].sum())
    top = (
        s.loc[s["is_anomaly"]]
        .nsmallest(20, "anomaly_score")
        .loc[:, ["timestamp", "temperature_2m", "relative_humidity_2m",
                 "surface_pressure", "wind_speed_10m", "anomaly_score"]]
        .copy()
    )
    top["timestamp"] = top["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    top = top.round(2)

    A: list[str] = []
    a = A.append
    a("# FASE 4 — Minería de datos: Detección de anomalías\n")
    a(f"**Ubicación:** {cfg.name} · **Método:** Isolation Forest sobre residuales "
      f"des-estacionalizados · **contamination = {model.contamination}** · "
      f"**Generado:** {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    a("> Climatología y modelo ajustados con *train* (2000–2018). Reproducible: "
      "`python -m backend.ml.mining.run_anomalies`.\n")
    a("\n---\n")

    a("## 1. Planteamiento\n")
    a("Aplicar Isolation Forest a la temperatura cruda marcaría casi todo el verano "
      "como anómalo (está lejos de la media anual). En su lugar se usan **residuales**:\n")
    a("```\nresidual = valor observado − media histórica de ese (mes, hora)\n```\n")
    a(f"Variables usadas: {', '.join(an.ANOMALY_VARIABLES)}. "
      "Así 'anómalo' = *inusual para esa época del año y hora del día*.\n")

    a("\n---\n")
    a("## 2. Resultados\n")
    a(f"- Observaciones puntuadas: **{len(s):,}**")
    a(f"- Marcadas como potencialmente anómalas: **{n_anom:,}** "
      f"({100 * n_anom / len(s):.2f} %)")
    a(f"- Umbral de score: {s.loc[s['is_anomaly'], 'anomaly_score'].max():.4f} "
      "(por debajo → anómalo)\n")
    if figures:
        a(f"![anomalías](assets/{figures[0]})\n")

    a("\n---\n")
    a("## 3. Las 20 horas más anómalas\n")
    a(top.to_markdown(index=False))

    a("\n---\n")
    a("## 4. Interpretación responsable\n")
    a("- Una anomalía estadística significa **\"valor inusual respecto al histórico\"**, "
      "**no** necesariamente un fenómeno climático extremo ni un error de medición.")
    a("- El modelo detecta combinaciones raras (p. ej. mucho calor con presión muy baja "
      "y viento fuerte a la vez), no valores altos aislados.")
    a("- El umbral `contamination` es una **decisión operativa** (se marca ~2 % de las "
      "horas), no una verdad: subirlo o bajarlo cambia cuántas se señalan.\n")
    a("\n**Q-gate:** ✅ detección sobre residuales des-estacionalizados; salida con "
      "score y clasificación normal/anómalo; comunicación con matices.\n")
    a("\n**Siguiente (fase 5):** Machine Learning — predicción de temperatura.\n")
    return "\n".join(A)


_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    p = argparse.ArgumentParser(description="Detección de anomalías (Isolation Forest).")
    p.add_argument("--contamination", type=float, default=an.DEFAULT_CONTAMINATION)
    args = p.parse_args()

    cfg = load_config()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    model = an.fit(train, contamination=args.contamination)

    with session_scope() as session:
        loc = get_location(session, cfg.slug)
        all_obs = load_observations_df(session, cfg.slug, with_id=True)
        scored = all_obs.join(model.score(all_obs))

        session.execute(delete(Anomaly).where(Anomaly.detector == DETECTOR))
        session.flush()

        valid = scored["anomaly_score"].notna()
        records = [
            {
                "location_id": loc.id,
                "observation_id": int(row.observation_id),
                "observed_at": row.timestamp.to_pydatetime(),
                "anomaly_score": round(float(row.anomaly_score), 5),
                "is_anomaly": bool(row.is_anomaly),
                "detector": DETECTOR,
                "features_used": list(an.ANOMALY_VARIABLES),
            }
            for row in scored.loc[valid].itertuples(index=False)
        ]
        bulk_upsert(session, Anomaly, records,
                    conflict_index=["observation_id", "detector"], label="anomalies")

    artifact = ARTIFACTS_DIR / f"iforest_{cfg.slug}.joblib"
    joblib.dump(
        {"iforest": model.iforest, "climatology": model.climatology,
         "variables": list(an.ANOMALY_VARIABLES), "contamination": model.contamination,
         "trained_at": datetime.now(UTC).isoformat()},
        artifact,
    )
    log.info("Artefacto: %s", artifact)

    figures = _figures(scored, cfg.slug)
    (DOCS_DIR / "fase-4-anomalias.md").write_text(_report(scored, model, figures), encoding="utf-8")
    log.info("Informe: docs/fase-4-anomalias.md | anomalías: %d de %d",
             int(scored["is_anomaly"].sum()), int(scored["anomaly_score"].notna().sum()))


if __name__ == "__main__":
    main()
