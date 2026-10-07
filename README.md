# Weather Intelligence

Plataforma de minería de datos, Machine Learning e IA generativa para el análisis
meteorológico de **Madrid-Barajas**, siguiendo la metodología **CRISP-ML(Q)**.

Este repositorio contiene el **backend** (API FastAPI, pipeline de datos, modelos
de ML, base de datos y documentación). El dashboard vive en su propio repo:
[sara9559ojeda/weather_intelligence_front](https://github.com/sara9559ojeda/weather_intelligence_front).

Proyecto final universitario. **Informe final (para la defensa):**
[`docs/fase-10-informe-final.md`](docs/fase-10-informe-final.md). Diseño
original: [`docs/fase-1-analisis-y-arquitectura.md`](docs/fase-1-analisis-y-arquitectura.md).

---

## Estado del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Análisis y arquitectura | ✅ Cerrada |
| 2 | Data Understanding + adquisición + esquema BD + ETL | ✅ Cerrada |
| 3 | Data Preparation + feature engineering + EDA | ✅ Cerrada |
| 4 | Minería de datos (clustering K-Means + anomalías Isolation Forest) | ✅ Cerrada |
| 5 | Machine Learning (predicción de temperatura multi-horizonte) | ✅ Cerrada |
| 6 | Backend API (FastAPI) + ingesta en tiempo real + SSE | ✅ Cerrada |
| 7 | Integración con Claude (capa de interpretación) | ✅ Cerrada |
| 8 | Frontend (React + Tailwind + Recharts) + dashboard | ✅ Cerrada |
| 9 | Retraining con criterio de promoción + robustez | ✅ Cerrada |
| 10 | Documentación académica | ✅ Cerrada |

---

## Requisitos

- Python 3.12
- Docker + Docker Compose (para PostgreSQL)

## Puesta en marcha

```bash
# 1. Variables de entorno
cp .env.example .env        # y edita los valores

# 2. Entorno Python (desde la raíz del repo)
python -m venv backend/.venv
backend/.venv/Scripts/pip install -r backend/requirements-dev.txt   # Windows
# source backend/.venv/bin/activate && pip install -r backend/requirements-dev.txt   # Linux/Mac

# 3. Base de datos
docker compose up -d db
cd backend && python -m alembic upgrade head && cd ..

# 4. Descarga de datos (~9 min la primera vez; con caché es instantáneo)
python -m backend.ml.acquisition.download_historical   # Open-Meteo / ERA5
python -m backend.ml.acquisition.download_meteostat    # estación real (contraste)

# 5. Perfilado + informe de Data Understanding
python -m backend.ml.data_understanding                # -> docs/fase-2-data-understanding.md

# 6. Carga a PostgreSQL (idempotente)
python -m backend.app.services.etl.load_historical     # -> tabla weather_observations

# 7. Preparación: limpieza + features + partición cronológica
python -m backend.ml.preparation.build_dataset         # -> data/processed/{train,valid,test}.parquet

# 8. Análisis exploratorio (EDA)
python -m backend.ml.eda                               # -> docs/fase-3-eda.md

# 9. Minería de datos
python -m backend.ml.mining.run_clustering             # K-Means -> cluster_models/cluster_assignments
python -m backend.ml.mining.run_anomalies              # Isolation Forest -> anomalies

# 10. Machine Learning: predicción de temperatura (~4 min)
python -m backend.ml.modeling.run_training             # -> model_runs, predictions, artefactos

# 11. API backend (con recarga en desarrollo)
backend/.venv/Scripts/uvicorn backend.app.main:app --reload   # http://localhost:8000/docs

# 12. Dashboard (en otra terminal) — repo aparte, clonado dentro de ./frontend
git clone https://github.com/sara9559ojeda/weather_intelligence_front.git frontend
npm --prefix frontend install
npm --prefix frontend run dev                          # http://localhost:5173

# 13. Reentrenamiento (fase 9) — normalmente lo dispara el scheduler solo
python -m backend.ml.modeling.run_retraining --check-only  # ¿toca reentrenar?
python -m backend.ml.modeling.run_retraining                # reentrena si toca
python -m backend.ml.modeling.run_retraining --force         # fuerza uno ahora (demo, ~5 min)
```

O todo junto con Docker (sirve la API y el dashboard contra la BD ya preparada):

```bash
git clone https://github.com/sara9559ojeda/weather_intelligence_front.git frontend   # si aún no está
docker compose up -d          # db + backend + frontend  ->  http://localhost:5173
```

Los datos van a `data/raw/` (ignorados por git); sus metadatos
(`*.meta.json`, con `sha256`, rango y nº de filas) **sí** se versionan.

## Pruebas

```bash
backend/.venv/Scripts/python -m pytest backend/tests -m "not network and not db"   # rápidas, sin dependencias
backend/.venv/Scripts/python -m pytest backend/tests -m "db"                        # requieren PostgreSQL en marcha
backend/.venv/Scripts/python -m pytest backend/tests                                # todas (red + BD)
```

Linter: `backend/.venv/Scripts/python -m ruff check backend/`

---

## Estructura

```
docs/              documentación académica (diseño, Data Understanding, diccionario de datos)
data/              raw / interim / processed  (datos ignorados por git)
backend/
  alembic/         migraciones de la base de datos
  Dockerfile
  app/
    main.py        app FastAPI (lifespan: backfill + scheduler), CORS, errores
    core/config.py configuración tipada leída de .env
    api/routes/    endpoints REST (health, weather, analytics, predictions, dashboard, stream)
    schemas/       modelos Pydantic de respuesta
    scheduler/     APScheduler: ingesta en tiempo real (30 min) + comprobación de reentrenamiento (24 h)
    db/            modelos SQLAlchemy, engine, sesión, consultas
    services/
      etl/         carga de data/raw -> PostgreSQL
      features/    construcción de features (compartido entre entrenamiento e inferencia)
      prediction/  inferencia: carga los artefactos y predice
      ingestion/   WeatherProvider (adapter) + Open-Meteo Forecast + ETL tiempo real
      historical/  comparación con el histórico (percentiles condicionados, régimen, anomalía)
      claude/      capa de interpretación (prompt + cliente Anthropic + caché + fallback)
      dashboard.py ensamblado de /dashboard y del resumen para Claude
  ml/
    config/        location.toml — fuente de verdad de ubicación y parámetros
    acquisition/   descarga de datos (Open-Meteo, Meteostat)
    preparation/   limpieza + partición cronológica + build_dataset
    mining/        clustering (K-Means) y anomalías (Isolation Forest)
    modeling/      baselines + Ridge/RF/HistGB + selección por horizonte + reentrenamiento (retrain.py)
    notebooks/     01_data_understanding … 04_modeling
    artifacts/     modelos serializados (.joblib)
    quality_rules.py        rangos físicos plausibles por variable
    data_understanding.py   perfilado + informe reproducible
    eda.py                  análisis exploratorio + informe
  tests/
docker-compose.yml
```

## Fuentes de datos y licencias

- **Histórico:** Open-Meteo Historical Weather API (reanálisis ERA5, ECMWF/Copernicus) — datos CC BY 4.0.
- **Contraste:** Meteostat, estación 08221 Madrid-Barajas (LEMD) — datos CC BY 4.0.
- **Tiempo real:** Open-Meteo Forecast API — datos CC BY 4.0.
- **Interpretación:** API de Anthropic (Claude). La clave vive solo en el backend.

## Interpretación con Claude (opcional)

Sin `ANTHROPIC_API_KEY` el sistema funciona igual: `GET /api/insights` devuelve el
resumen estructurado sin el texto interpretado. Para activarlo, añade la clave a
`.env` del backend y reinicia:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

## Reentrenamiento (fase 9)

El scheduler comprueba cada `RETRAIN_CHECK_INTERVAL_HOURS` (24 h por defecto)
si toca reentrenar — por volumen de observaciones nuevas (`RETRAIN_VOLUME_TRIGGER`,
1000) o por calendario (`RETRAIN_CALENDAR_DAYS`, 30 días). El modelo nuevo
**solo sustituye** al activo si mejora su RMSE al menos `RETRAIN_EPSILON` (2 %)
sobre el mismo conjunto de test, o si el activo ha degradado más de
`RETRAIN_DEGRADATION_THRESHOLD` (20 %). Todo intento (se promocione o no)
queda en `retraining_runs` — `GET /api/model/retraining-history`. Detalle y
una ejecución real verificada en
[`docs/fase-9-retraining.md`](docs/fase-9-retraining.md).

## Reproducibilidad

- Semillas fijas (`random_state=42`) en todos los modelos y en el muestreo.
- Versiones ancladas: `backend/requirements.txt` (Python), `package.json` del repo del frontend
  (Node, con `package-lock.json`).
- Cada dataset crudo lleva un `*.meta.json` con `sha256`, fuente, licencia y
  rango — `data/raw/` y `data/processed/*.json` documentan de dónde sale cada
  número que aparece en los informes de `docs/`.
- La partición train/valid/test es cronológica y sus límites exactos quedan en
  `data/processed/split_meta.json` — no cambian entre ejecuciones a menos que
  cambien los datos de entrada.
- Todo *pipeline* (adquisición, ETL, preparación, minería, modelado,
  reentrenamiento) es un módulo de `backend/ml/` ejecutable por separado con
  `python -m`, con *logging* y sin estado oculto entre pasos.
