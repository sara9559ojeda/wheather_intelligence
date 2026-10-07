"""CLI: clustering K-Means de regímenes meteorológicos (fase 4).

    - ajusta el modelo con el tramo de TRAIN
    - elige k con Elbow + Silhouette
    - interpreta cada cluster (centroide en unidades reales + etiqueta)
    - asigna TODAS las observaciones y persiste en cluster_models / cluster_assignments
    - guarda el artefacto y genera docs/fase-4-clustering.md + figuras

Uso:
    python -m backend.ml.mining.run_clustering
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import delete

from backend.app.db.bulk import bulk_upsert
from backend.app.db.models import ClusterAssignment, ClusterModel
from backend.app.db.queries import load_observations_df
from backend.app.db.session import session_scope
from backend.ml.config_loader import ARTIFACTS_DIR, PROCESSED_DIR, PROJECT_ROOT, load_config
from backend.ml.mining import clustering as clu

log = logging.getLogger("mining.clustering")
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"


def _figures(scores: pd.DataFrame, model: clu.ClusterModel, df_train: pd.DataFrame,
             assign_all: pd.DataFrame, slug: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. Elbow + Silhouette
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(scores["k"], scores["inertia"], "o-", color="#3b7dd8", label="inertia")
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia (SSE)", color="#3b7dd8")
    ax2 = ax1.twinx()
    ax2.plot(scores["k"], scores["silhouette"], "s-", color="#d8663b", label="silhouette")
    ax2.set_ylabel("silhouette medio", color="#d8663b")
    ax1.axvline(model.k, ls="--", color="gray")
    ax1.set_title("Selección de k: método del codo y silhouette")
    fig.tight_layout()
    p = ASSETS_DIR / f"f4_{slug}_seleccion_k.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 2. Proyección PCA 2D de una muestra, coloreada por cluster
    X = clu.build_matrix(df_train)
    xs = model.scaler.transform(X)
    rng = np.random.default_rng(clu.RANDOM_STATE)
    idx = rng.choice(len(xs), min(8000, len(xs)), replace=False)
    pca = PCA(n_components=2, random_state=clu.RANDOM_STATE).fit(xs)
    proj = pca.transform(xs[idx])
    labels = model.kmeans.predict(xs[idx])
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=6, alpha=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Clusters proyectados (PCA, {pca.explained_variance_ratio_.sum():.0%} var.)")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    p = ASSETS_DIR / f"f4_{slug}_pca.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 3. Distribución de cada cluster por mes y por hora
    a = assign_all.copy()
    a["month"] = a["timestamp"].dt.month
    a["hour"] = a["timestamp"].dt.hour
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for by, ax in (("month", axes[0]), ("hour", axes[1])):
        tab = pd.crosstab(a[by], a["cluster_id"], normalize="index")
        tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.9,
                 legend=(by == "hour"))
        ax.set_title(f"Composición de clusters por {by}")
        ax.set_ylabel("proporción")
    fig.tight_layout()
    p = ASSETS_DIR / f"f4_{slug}_calendario.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)
    return made


def _report(sel: clu.KSelection, interp: pd.DataFrame, model: clu.ClusterModel,
            assign_all: pd.DataFrame, figures: list[str]) -> str:
    cfg = load_config()
    out: list[str] = []
    A = out.append
    A("# FASE 4 — Minería de datos: Clustering de regímenes meteorológicos\n")
    A(f"**Ubicación:** {cfg.name} · **Algoritmo:** K-Means · **k = {model.k}** · "
      f"**Generado:** {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    A("> Modelo ajustado con el tramo de *train* (2000–2018). Reproducible: "
      "`python -m backend.ml.mining.run_clustering`.\n")
    A("\n---\n")
    A("## 1. Selección del número de clusters\n")
    A(sel.reason + "\n")
    A(sel.scores.round(4).to_markdown(index=False))
    A(f"\n![selección de k](assets/f4_{cfg.slug}_seleccion_k.png)\n")
    A("\n> El silhouette es modesto (los datos meteorológicos son un continuo, no "
      "grupos bien separados), pero los clusters resultantes son **físicamente "
      "interpretables**, que es el criterio que importa aquí.\n")

    A("\n---\n")
    A("## 2. Interpretación de los clusters\n")
    A("Centroides en **unidades originales** (precipitación en mm tras deshacer el log):\n")
    A(interp.to_markdown(index=False))
    A("\n**Etiquetas** (derivadas de la desviación del centroide respecto a la media global, "
      "no fijadas de antemano):\n")
    for _, r in interp.iterrows():
        A(f"- **Cluster {r['cluster']}** ({r['pct']} % de las horas): "
          f"*{r['etiqueta']}* — {r['temperature_2m']} °C, "
          f"{r['relative_humidity_2m']} % HR, {r['cloud_cover']} % nubes, "
          f"{r['shortwave_radiation']} W/m².")

    A("\n---\n")
    A("## 3. Distribución temporal\n")
    A(f"![pca](assets/f4_{cfg.slug}_pca.png)\n")
    A(f"![calendario](assets/f4_{cfg.slug}_calendario.png)\n")
    A("\nCada régimen aparece con mayor o menor frecuencia según el mes y la hora "
      "(p. ej. los regímenes cálidos y secos dominan en verano y de día). Esto es "
      "una **consecuencia** de los datos, no una entrada del modelo: el clustering "
      "no vio la fecha ni la hora.\n")

    A("\n---\n")
    A("## 4. Persistencia\n")
    A(f"- `cluster_models`: 1 fila (k={model.k}, silhouette="
      f"{sel.scores.loc[sel.scores['k'] == model.k, 'silhouette'].iloc[0]:.4f}, is_active=true)")
    A(f"- `cluster_assignments`: {len(assign_all):,} filas "
      "(una por observación: cluster + distancia al centroide)")
    A(f"- artefacto: `backend/ml/artifacts/kmeans_{cfg.slug}.joblib`\n")

    A("\n**Q-gate:** ✅ k elegido con Elbow+Silhouette y documentado; clusters "
      "interpretados desde los datos; asignaciones persistidas.\n")
    A("\n**Siguiente:** detección de anomalías (Isolation Forest sobre residuales).\n")
    return "\n".join(out)


_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    cfg = load_config()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    log.info("Train: %d filas", len(train))

    X = clu.build_matrix(train)
    x_scaled = StandardScaler().fit_transform(X)

    log.info("Evaluando k de 2 a 10…")
    scores = clu.evaluate_k(x_scaled)
    sel = clu.choose_k(scores)
    log.info("k elegido: %d — %s", sel.chosen_k, sel.reason)

    model = clu.fit(train, sel.chosen_k)
    interp = clu.interpret(model, train)
    log.info("Clusters:\n%s", interp.to_string(index=False))

    with session_scope() as session:
        all_obs = load_observations_df(session, cfg.slug, with_id=True)
        labels, dist = model.assign(all_obs)
        valid = ~np.isnan(dist)
        assign_all = pd.DataFrame({
            "observation_id": all_obs.loc[valid, "observation_id"].to_numpy(),
            "timestamp": all_obs.loc[valid, "timestamp"].to_numpy(),
            "cluster_id": labels[valid],
            "distance_to_centroid": dist[valid],
        })

        session.execute(delete(ClusterModel))  # empezar de cero (versión anterior en el artefacto)
        session.flush()
        sil = float(scores.loc[scores["k"] == model.k, "silhouette"].iloc[0])
        cm = ClusterModel(
            k=model.k,
            silhouette=sil,
            feature_list=list(model.features),
            centroids=json.loads(interp.to_json(orient="records")),
            cluster_labels={int(r["cluster"]): r["etiqueta"] for _, r in interp.iterrows()},
            artifact_path=f"backend/ml/artifacts/kmeans_{cfg.slug}.joblib",
            is_active=True,
        )
        session.add(cm)
        session.flush()

        records = [
            {
                "cluster_model_id": cm.id,
                "observation_id": int(row.observation_id),
                "cluster_id": int(row.cluster_id),
                "distance_to_centroid": float(row.distance_to_centroid),
            }
            for row in assign_all.itertuples(index=False)
        ]
        bulk_upsert(session, ClusterAssignment, records,
                    conflict_index=["cluster_model_id", "observation_id"],
                    label="cluster_assignments")

    artifact = ARTIFACTS_DIR / f"kmeans_{cfg.slug}.joblib"
    joblib.dump(
        {"scaler": model.scaler, "kmeans": model.kmeans, "features": model.features,
         "k": model.k, "trained_at": datetime.now(UTC).isoformat()},
        artifact,
    )
    log.info("Artefacto: %s", artifact)

    figures = _figures(scores, model, train, assign_all, cfg.slug)
    (DOCS_DIR / "fase-4-clustering.md").write_text(
        _report(sel, interp, model, assign_all, figures), encoding="utf-8"
    )
    log.info("Informe: docs/fase-4-clustering.md")


if __name__ == "__main__":
    main()
