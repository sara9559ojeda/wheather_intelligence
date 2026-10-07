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
# # 04 — Machine Learning: predicción de temperatura (fase 5)
#
# Reproduce y visualiza lo que hace `python -m backend.ml.modeling.run_training`:
# comparar baselines + Ridge/RandomForest/HistGradientBoosting a 5 horizontes,
# con partición cronológica y validación cruzada temporal.

# %%
import pandas as pd

from backend.app.services.features.builder import HORIZONS_HOURS, target_column
from backend.ml.config_loader import PROCESSED_DIR
from backend.ml.modeling.baselines import SeasonalBaseline
from backend.ml.modeling.train import train_horizon

train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
manifest = pd.read_json(PROCESSED_DIR / "dataset_manifest.json", typ="series")
feature_cols = list(manifest["feature_columns"])
len(feature_cols), len(train), len(valid), len(test)

# %% [markdown]
# ## Entrenar y comparar por horizonte
#
# (cada horizonte tarda ~30 s por los ajustes de Random Forest)

# %%
seasonal = SeasonalBaseline().fit(train)
rows = []
results = {}
for h in HORIZONS_HOURS:
    res = train_horizon(train, valid, test, feature_cols, h, seasonal)
    results[h] = res
    rows.append({
        "H": h,
        "persistencia": res.validation["baseline_persistence"]["rmse"],
        "climatología": res.validation["baseline_seasonal"]["rmse"],
        "ridge": res.validation["ridge"]["rmse"],
        "random_forest": res.validation["random_forest"]["rmse"],
        "hist_gb": res.validation["hist_gradient_boosting"]["rmse"],
        "campeón": res.champion_type,
        "test_rmse": res.test_champion["rmse"],
        "skill_vs_persist": res.test_champion["skill_vs_persistence"],
    })
pd.DataFrame(rows)

# %% [markdown]
# ## Predicho vs real — campeón de H=6h sobre las últimas 2 semanas de test

# %%
r = results[6]
tcol = target_column(6)
te = test.dropna(subset=[tcol]).tail(24 * 14)
pred = r.champion.predict(te[feature_cols])

ax = te.set_index("timestamp")[tcol].plot(figsize=(12, 4), label="real")
pd.Series(pred, index=te["timestamp"]).plot(ax=ax, label="predicción +6h")
ax.fill_between(te["timestamp"], pred + r.residual_p05, pred + r.residual_p95, alpha=0.2)
ax.legend()
ax.set_title("H=6h — predicho vs real")

# %% [markdown]
# ## Importancia de variables del campeón (H=6h)

# %%
pd.Series(r.feature_importance).sort_values(ascending=False).head(15)

# %% [markdown]
# Los scripts persisten además: 25 filas en `model_runs`, ~175 000 en
# `predictions`, y los artefactos `backend/ml/artifacts/model_temp_h{H}.joblib`
# que carga el servicio de inferencia (`backend.app.services.prediction`).
