# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Minería de datos (fase 4)
#
# Clustering de regímenes meteorológicos (K-Means) y detección de anomalías
# (Isolation Forest sobre residuales des-estacionalizados).
#
# Este notebook **reproduce y visualiza** lo que ya hacen los scripts:
#
# ```
# python -m backend.ml.mining.run_clustering
# python -m backend.ml.mining.run_anomalies
# ```
#
# Ejecutar desde la raíz del repo con el intérprete de `backend/.venv`.

# %%
import joblib
import pandas as pd

from backend.ml.config_loader import ARTIFACTS_DIR, PROCESSED_DIR, load_config
from backend.ml.mining import anomalies as an
from backend.ml.mining import clustering as clu

cfg = load_config()
train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
train.shape

# %% [markdown]
# ## 1. Clustering — elección de k

# %%
from sklearn.preprocessing import StandardScaler

X = clu.build_matrix(train)
x_scaled = StandardScaler().fit_transform(X)
scores = clu.evaluate_k(x_scaled)
scores

# %%
sel = clu.choose_k(scores)
print(sel.reason)
model = clu.fit(train, sel.chosen_k)
interp = clu.interpret(model, train)
interp

# %% [markdown]
# Cada cluster es un **régimen meteorológico** que surge de los datos
# (el modelo no vio la hora ni el mes). Las etiquetas se derivan de la
# desviación del centroide respecto a la media global.

# %% [markdown]
# ## 2. Anomalías — Isolation Forest sobre residuales

# %%
amodel = an.fit(train, contamination=0.02)
clim = amodel.climatology
clim.head()

# %%
scored = train.join(amodel.score(train))
print("anómalas:", int(scored["is_anomaly"].sum()), "de", len(scored))
scored.loc[scored["is_anomaly"]].nsmallest(10, "anomaly_score")[
    ["timestamp", "temperature_2m", "relative_humidity_2m", "surface_pressure",
     "wind_speed_10m", "anomaly_score"]
]

# %% [markdown]
# ## 3. Artefactos persistidos
#
# Los scripts guardan además:
#
# - `backend/ml/artifacts/kmeans_<slug>.joblib`
# - `backend/ml/artifacts/iforest_<slug>.joblib`
#
# y escriben las asignaciones en las tablas `cluster_assignments` y `anomalies`.

# %%
kmeans_art = joblib.load(ARTIFACTS_DIR / f"kmeans_{cfg.slug}.joblib")
iforest_art = joblib.load(ARTIFACTS_DIR / f"iforest_{cfg.slug}.joblib")
kmeans_art["k"], iforest_art["contamination"]
