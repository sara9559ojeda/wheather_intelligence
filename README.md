# Weather Intelligence

Plataforma de minería de datos, Machine Learning e IA generativa para el análisis
meteorológico de **varias ciudades** (Madrid-Barajas y Bogotá, Medellín, Cali, Barranquilla y
Cartagena), siguiendo la metodología **CRISP-ML(Q)**. Cada ciudad tiene sus propios datos,
modelos, regímenes y anomalías, y se elige desde un selector en la app.

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
| 11 | Inteligencia contextual y explicable (normalidad, regímenes, evaluación real, confiabilidad, analista) | ✅ Implementada — [`docs/fase-11-inteligencia-contextual.md`](docs/fase-11-inteligencia-contextual.md) |

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

# 10b. Más ciudades (opcional): descarga, entrena y publica una ciudad del catálogo
#      (backend/ml/config/locations.toml). Tarda ~10-15 min por ciudad por la descarga ERA5.
python -m backend.ml.add_location --slug bogota        # o --all para todas las del catálogo
#      Al terminar aparece en GET /api/locations y en el selector de la app.
#      Para añadir otra ciudad basta un bloque [[locations]] nuevo en locations.toml.

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
docker compose up -d          # db + redis + backend + frontend  ->  http://localhost:5173
```

Los datos van a `data/raw/` (ignorados por git); sus metadatos
(`*.meta.json`, con `sha256`, rango y nº de filas) **sí** se versionan.

## Pruebas

```bash
backend/.venv/Scripts/python -m pytest backend/tests -m "not network and not db and not redis"   # rápidas
backend/.venv/Scripts/python -m pytest backend/tests -m "db"      # requieren PostgreSQL (docker compose up -d db)
backend/.venv/Scripts/python -m pytest backend/tests -m "redis"   # requieren Redis (docker compose up -d redis)
backend/.venv/Scripts/python -m pytest backend/tests              # todas (red + BD + Redis)
```

Linter: `backend/.venv/Scripts/python -m ruff check backend/`

---

## Estructura

```
docs/              documentación académica (diseño, Data Understanding, diccionario de datos)
requirements.txt   dependencias de runtime de la API (Docker y Vercel)
index.py           entrypoint de Vercel (reexporta backend.app.main:app)
vercel.json        configuración de la función de Vercel (duración, ficheros incluidos/excluidos)
.github/workflows/ ingest.yml — dispara la ingesta cada 30 min en el despliegue serverless
data/              raw / interim / processed  (datos ignorados por git)
backend/
  alembic/         migraciones de la base de datos
  Dockerfile
  app/
    main.py        app FastAPI (lifespan: backfill + scheduler), CORS, errores
    core/config.py configuración tipada leída de .env
    api/routes/    endpoints REST (health, auth, weather, analytics, predictions, dashboard, ...),
                   tiempo real (ws = WebSocket, stream = SSE) y tasks (POST /api/tasks/ingest)
    schemas/       modelos Pydantic de respuesta
    scheduler/     APScheduler: ingesta en tiempo real (30 min) + comprobación de reentrenamiento (24 h)
    db/            modelos SQLAlchemy, engine, sesión, consultas
    services/
      etl/         carga de data/raw -> PostgreSQL
      features/    construcción de features (compartido entre entrenamiento e inferencia)
      prediction/  inferencia: carga los artefactos y predice
      ingestion/   WeatherProvider (adapter) + Open-Meteo Forecast + ETL tiempo real
      historical/  comparación con el histórico (percentiles condicionados, régimen, anomalía)
      analysis/    normalidad histórica (F1) y regímenes (F2), por ciudad
      evaluation/  registro y verificación de pronósticos (F3), calibración e intervalos (F4)
      analyst/     analista con respaldo determinista, verificación numérica y límites (F5)
      claude/      capa de interpretación (prompt + cliente Anthropic + caché + fallback)
      dashboard.py ensamblado de /dashboard y del resumen para Claude
      events.py    bus de eventos de tiempo real (memoria o Redis pub/sub)
  requirements.txt  pipelines offline: incluye ../requirements.txt + pyarrow, meteostat
  ml/
    config/        location.toml (Madrid + parámetros comunes) y locations.toml (catálogo de ciudades)
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

## Inteligencia contextual (fase 11)

Cinco módulos por ciudad, en las pestañas **Análisis histórico**, **Inteligencia predictiva**
y **Analista IA** del dashboard:

| Módulo | Endpoint |
|---|---|
| ¿Qué tan normal es el tiempo de hoy? | `GET /api/analysis/normality` |
| Patrones (regímenes) de la ciudad | `GET /api/analysis/regimes` |
| ¿Qué tan bien predice Weather Intelligence? | `GET /api/evaluation/summary`, `/api/evaluation/pairs` |
| ¿Qué tan confiable es el pronóstico? | `GET /api/forecast/reliability` |
| Pregúntale a Weather Intelligence | `GET /api/analyst/questions`, `POST /api/analyst/ask` |

Todos exigen sesión y aceptan `?location=<slug>`. Cuando no hay datos suficientes lo dicen
(«preliminar», «el registro acaba de empezar») en vez de inventar métricas: el registro de
pronósticos reales empezó el 8 nov 2026 y las métricas aparecen conforme se verifican.
El «observado» es reanálisis/modelo de Open-Meteo, **no** una estación. El analista funciona sin
`ANTHROPIC_API_KEY` (respuestas deterministas con los datos reales); con clave, Claude solo recibe
el contexto estructurado —nunca credenciales ni tokens— y cualquier cifra que no esté respaldada
descarta su texto. Ajustes opcionales: `ANALYST_TIMEOUT_SECONDS`, `ANALYST_AI_PER_HOUR`,
`ANALYST_ASKS_PER_HOUR`, `ANALYST_CACHE_MINUTES`. Para evaluar offline la cobertura de los
intervalos: `python -m backend.ml.modeling.evaluate_intervals --all`.
Métodos, límites, hallazgos y pruebas: [`docs/fase-11-inteligencia-contextual.md`](docs/fase-11-inteligencia-contextual.md).

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

## Tiempo real y despliegue en Vercel

El dashboard recibe avisos en tiempo real por **WebSocket** (`ws://…/api/ws`) o
por SSE (`/api/stream`, se mantiene mientras el frontend migra). Ambos salen del
mismo bus de eventos: en memoria si `REDIS_URL` está vacío, **Redis pub/sub** si
no (Docker Compose levanta `redis:7-alpine`; en Vercel, Upstash). Mensajes del
WebSocket (JSON): `{"type": "connected"|"observation"|"model_updated"|"forecasts_verified", "data": {...}}`
y `{"type": "ping"}` cada 25 s sin tráfico; el cliente debe reconectar con
*backoff* al cerrarse la conexión. Contrato completo en
[`docs/fase-6-backend-api.md` §9](docs/fase-6-backend-api.md).

El backend se puede desplegar en **Vercel Functions** (entrypoint `index.py`,
`vercel.json`, `requirements.txt` de la raíz). En serverless no hay scheduler:
`ENABLE_SCHEDULER=false`, `RUN_STARTUP_BACKFILL=false`, `DB_POOL_MODE=null`, y la
ingesta de cada 30 min la dispara [`.github/workflows/ingest.yml`](.github/workflows/ingest.yml)
contra `POST /api/tasks/ingest` (`Authorization: Bearer $INGEST_TOKEN`; sin
token configurado el endpoint queda deshabilitado). Las migraciones se aplican a
mano (`alembic upgrade head`) contra la URL directa de Neon.

## Despliegue en Railway (recomendado)

Un solo servicio con la imagen de `backend/Dockerfile` (lo configura
[`railway.toml`](railway.toml)): API, scheduler de ingesta cada 30 min y
reentrenamiento en el mismo proceso, con un Postgres de Railway. Las migraciones
se aplican al arrancar. Sin Redis, sin GitHub Actions y sin Neon: el bus de
eventos en memoria basta con **una sola réplica** (no subas `numReplicas`).

1. Proyecto nuevo en Railway → **Add PostgreSQL**.
2. Cargar los datos locales (antes de desplegar el backend). Con la
   `DATABASE_PUBLIC_URL` del Postgres (pestaña *Variables*):
   ```bash
   docker exec wi_db pg_dump -U weather -d weather_intelligence -Fc --no-owner --no-privileges -f /tmp/wi.dump
   docker cp wi_db:/tmp/wi.dump ./wi.dump
   docker run --rm -v "$PWD:/d" postgres:16-alpine pg_restore --no-owner --no-privileges --clean --if-exists -d "<DATABASE_PUBLIC_URL>" /d/wi.dump
   ```
3. **New service → GitHub repo** (`wheather_intelligence`). Variables:

   | Variable | Valor |
   |---|---|
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (referencia a la red privada) |
   | `AUTH_SECRET_KEY` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `ALLOW_REGISTRATION` | `false` (las cuentas vienen en la copia de la BD) |
   | `ANTHROPIC_API_KEY` | opcional |
   | `BACKEND_CORS_ORIGINS` | dominio del frontend, p. ej. `https://<app>.vercel.app` |

   El resto va con los valores por defecto (`ENABLE_SCHEDULER=true`,
   `RUN_STARTUP_BACKFILL=true`, `DB_POOL_MODE=queue`, `REDIS_URL` vacío).
4. *Settings → Networking → Generate Domain* y poner esa URL como
   `VITE_API_URL` en el proyecto del frontend (Vercel) y volver a desplegarlo.

## Login

Toda la API de datos exige sesión (JWT); el dashboard abre en una pantalla de
inicio de sesión / registro. Necesita `AUTH_SECRET_KEY` en `.env` (sin él, la API
responde 503 en vez de quedar abierta):

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"   # -> AUTH_SECRET_KEY=...
```

El registro público viene activado (`ALLOW_REGISTRATION=true`). Para cerrarlo y
dar de alta cuentas a mano:

```bash
docker compose exec backend python -m backend.app.create_user ana@example.com --name "Ana"
```

Contraseñas con scrypt, bloqueo tras 5 intentos fallidos, y tickets de 60 s para
el WebSocket. Detalle y decisiones en [`docs/fase-6-backend-api.md` §10](docs/fase-6-backend-api.md).

## Reproducibilidad

- Semillas fijas (`random_state=42`) en todos los modelos y en el muestreo.
- Versiones ancladas: `requirements.txt` (runtime de la API) y `backend/requirements.txt`
  (pipelines offline, lo incluye) para Python; `package.json` del repo del frontend
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
