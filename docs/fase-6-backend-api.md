# FASE 6 — Backend API (FastAPI)

**Ubicación:** Madrid-Barajas · **Fecha:** 2026-09-08
**Arranque:** `docker compose up -d` · **Docs:** http://localhost:8000/docs

---

## 1. Qué expone

API REST documentada con **OpenAPI 3.1** (Swagger en `/docs`). Todo bajo `/api`.

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y de la BD |
| `GET /meta` | Metadatos: nº de observaciones históricas/tiempo real, última observación |
| `GET /weather/current` | Última observación + antigüedad + si va con retraso |
| `GET /weather/history?hours=&source=` | Serie de observaciones en una ventana |
| `GET /stats?days=` | Media, mediana, desviación y percentiles de las variables clave |
| `GET /anomalies?limit=&only_flagged=` | Registros de anomalías (Isolation Forest) |
| `GET /clusters` | Los 3 regímenes K-Means: etiqueta, % de horas, centroide |
| `GET /predictions/latest` | Predicción **en vivo** a 1/3/6/12/24 h desde las observaciones recientes |
| `GET /predictions/history?horizon_hours=` | Predicciones persistidas (para el gráfico predicho-vs-real) |
| `GET /model/runs` · `/model/performance` | Métricas de los 25 `model_runs` / de los 5 campeones |
| `GET /dashboard` | **Todo lo anterior en una sola respuesta** para el frontend |
| `GET /comparison/latest` | Comparación histórica de la última observación (percentiles + régimen + anomalía) |
| `GET /insights` | Resumen estructurado para Claude (la interpretación real llega en la fase 7) |
| `GET /stream` | **Server-Sent Events**: avisa cuando entra una observación nueva |
| `GET /ws` (WebSocket) | Mismos avisos por **WebSocket** — canal preferente desde la §9 |
| `POST /tasks/ingest` | Dispara una ingesta (modo serverless). Protegido con `Bearer INGEST_TOKEN` (§9) |

## 2. Ingesta en tiempo real

```
APScheduler (cada 30 min)
   -> Open-Meteo Forecast API (endpoint `current`)
   -> validación (recorte a rangos físicos)
   -> alineado a la hora + upsert en weather_observations (source='realtime')
   -> features (mismo código que el entrenamiento) -> predicción a 5 horizontes
   -> anomalía (Isolation Forest) + régimen (K-Means) + percentiles condicionados
   -> persistencia (predictions, anomalies)
   -> evento SSE -> el dashboard hace refetch
```

- **Adaptador de proveedor** (`WeatherProvider`, patrón *adapter*): cambiar de
  Open-Meteo a otra API = una clase nueva, no tocar el resto.
- **Al arrancar**: *backfill* de las últimas 72 h (para que el dashboard no esté
  vacío) + una ingesta de condiciones actuales.
- **Idempotente**: `ON CONFLICT` sobre `(location_id, observed_at, source)`; la
  observación de la hora en curso se refresca con la última lectura.
- **Alineado a la hora**: el `current` de Open-Meteo viene con marca de 15 min;
  se redondea a la hora para encajar en la rejilla horaria que espera el modelo.

## 3. Transporte hacia el dashboard: SSE

Elegido frente a *polling* y WebSockets (fase 1, §16): el dashboard solo
**recibe** avisos. Patrón **"SSE notifica, REST trae los datos"** — el evento
solo dice *qué* cambió; el frontend hace *refetch* del endpoint correspondiente.
Keepalive (`ping`) cada 20 s; reconexión automática del navegador.

## 4. Comparación histórica (fase 1, §15)

Para una observación, `comparison.py` calcula:

- **Percentil de cada variable condicionado** a mes ±1 y hora ±1 (comparar
  septiembre contra septiembres, no contra todo el año — corrección C1).
- **Régimen** (cluster K-Means más cercano) + distancia al centroide.
- **Puntuación de anomalía** (Isolation Forest sobre residuales
  des-estacionalizados).

Ejemplo real (`GET /comparison/latest`):

```json
{
  "percentiles": [
    {"variable": "temperature_2m", "value": 21.7, "percentile": 79.7,
     "normal_range": [10.8, 23.5]},
    {"variable": "wind_speed_10m", "value": 3.3, "percentile": 10.6, ...}
  ],
  "cluster_id": 2, "cluster_label": "frío, húmedo",
  "is_anomaly": false, "anomaly_score": -0.403
}
```

## 5. Seguridad y robustez

- **Secretos solo en el backend**: `ANTHROPIC_API_KEY` en `.env`, nunca en las
  respuestas ni en el frontend. Open-Meteo no necesita clave.
- **CORS** restringido a `BACKEND_CORS_ORIGINS`, solo métodos GET.
- **Validación** de entrada con Pydantic (`Query` con rangos y `pattern`).
- **Manejo de errores**: *handler* global que registra el *traceback* y devuelve
  un 500 genérico; 404/503 tipados donde corresponde.
- **Los artefactos de ML pueden faltar**: los endpoints responden 503, no
  rompen toda la API.
- **El job del scheduler nunca tumba el servicio**: captura toda excepción y
  hace *rollback*.
- **Logging** estructurado con nivel configurable (`LOG_LEVEL`).

## 6. Arquitectura del código

```
backend/app/
  main.py              app FastAPI, lifespan (backfill + scheduler), CORS, errores
  core/config.py       configuración tipada (.env)
  api/
    deps.py            get_db, get_location_slug
    routes/            health, weather, analytics, predictions, dashboard, insights,
                       stream (SSE), ws (WebSocket), tasks (POST /tasks/ingest)
  schemas/models.py    modelos Pydantic de respuesta
  services/
    ingestion/         WeatherProvider (adapter) + open_meteo + realtime
    historical/        comparison — percentiles condicionados, régimen, anomalía
    prediction/        TemperaturePredictor (carga artefactos, predice)
    artifacts.py       carga perezosa y cacheada de los .joblib
    events.py          bus de eventos (memoria o Redis pub/sub) para WebSocket y SSE
    dashboard.py       ensamblado del /dashboard y del resumen para Claude
  scheduler/jobs.py    APScheduler: job de ingesta cada 30 min (run_ingest reutilizable)
```

## 7. Empaquetado

`docker-compose.yml` — servicios `db` (PostgreSQL 16) y `backend`. El backend
aplica las migraciones Alembic al arrancar y luego levanta Uvicorn. Los
artefactos `.joblib` van dentro de la imagen.

> Nota: la imagen sirve la API; el *pipeline* de datos y ML (pasos 4–10 del
> README) se ejecuta una vez para poblar la BD y generar los artefactos.

## 8. Pruebas

`backend/tests/test_api.py` (13, con `TestClient`) y `test_ingestion.py` (3).
Marcador `db`. **48 pruebas** en total en verde.

## 9. Ampliación: WebSocket + Redis y despliegue serverless (Vercel)

*Añadido el 2026-11-04.* El backend pasa a poder desplegarse en **Vercel
Functions** (Fluid compute), con Postgres en **Neon** y Redis en **Upstash**.
Docker/local siguen funcionando exactamente igual.

### 9.1 Bus de eventos con dos backends

`services/events.py` mantiene la misma interfaz (`publish`, `subscribe`,
`unsubscribe`) y los mismos eventos (`observation`, `model_updated`), con dos
implementaciones según `REDIS_URL`:

| `REDIS_URL` | Backend | Cuándo |
|---|---|---|
| vacío | memoria (una cola por cliente, en proceso) | desarrollo local, tests |
| `redis://` / `rediss://` | **Redis pub/sub**, canal `weather_intelligence:events` | Docker Compose (servicio `redis`), Vercel + Upstash |

En Vercel cada conexión WebSocket queda **fijada a una instancia** y el evento
puede generarse en otra (la invocación de `POST /api/tasks/ingest`), así que el
reparto entre instancias lo hace Redis: `publish` hace `PUBLISH` (cliente
síncrono, porque se llama desde hilos) y cada cliente conectado abre su propia
suscripción con `redis.asyncio`. Si Redis se cae, la suscripción avisa y el
servidor cierra la conexión (1011) para que el cliente reconecte.

### 9.2 Contrato del WebSocket `GET /api/ws`

Frames de texto JSON, solo servidor → cliente (lo que mande el cliente se ignora):

| `type` | `data` | Qué debe hacer el cliente |
|---|---|---|
| `connected` | `{"heartbeat_seconds": 25, "max_connection_seconds": 0}` | Primer mensaje; la suscripción ya está activa. Hacer un refetch inicial |
| `observation` | `{"observed_at": "2026-11-05T04:00:00+00:00", "temperature_2m": 2.5, "is_anomaly": false}` | Refetch de `/api/dashboard` |
| `model_updated` | `{"status": "promoted", "promoted_horizons": ["1", "24"]}` | Refetch de `/api/model/*` |
| `ping` | — (sin campo `data`) | Latido si no hubo otro mensaje en 25 s; no requiere respuesta |

Tipos desconocidos: ignorarlos (el contrato puede crecer).

**Cierre y reconexión.** En Vercel la conexión se corta al llegar a la duración
máxima de la función (300 s en Hobby). Con `WS_MAX_CONNECTION_SECONDS=280` el
servidor la cierra antes, de forma ordenada, con código **1012**; con **1011**
si falla el bus. El cliente debe reconectar siempre con *backoff* exponencial
(1 s → 30 s) y hacer refetch al reconectar: los eventos emitidos mientras
estaba desconectado no se reenvían.

```js
// cliente mínimo (navegador)
function connect(delay = 1000) {
  const ws = new WebSocket(`${API_URL.replace(/^http/, "ws")}/api/ws`);
  ws.onopen = () => { delay = 1000; };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "connected" || msg.type === "observation") refetchDashboard();
    else if (msg.type === "model_updated") refetchModels();
  };
  ws.onclose = () => setTimeout(() => connect(Math.min(delay * 2, 30000)), delay);
}
```

`GET /api/stream` (SSE) se mantiene, alimentado por el mismo bus y con el mismo
formato que antes (`event: update`, `data: {"event": ..., "data": ...}`),
hasta que el frontend migre al WebSocket.

### 9.3 Modo serverless

| Variable | Docker/local | Vercel | Efecto |
|---|---|---|---|
| `ENABLE_SCHEDULER` | `true` | `false` | APScheduler (ingesta 30 min + comprobación de reentreno) |
| `RUN_STARTUP_BACKFILL` | `true` | `false` | backfill de 72 h + ingesta al arrancar |
| `DB_POOL_MODE` | `queue` | `null` | `NullPool` (sin conexiones retenidas; el pooling lo hace el pooler de Neon) |
| `REDIS_URL` | `redis://redis:6379/0` (compose) | URL `rediss://` de Upstash | bus de eventos |
| `INGEST_TOKEN` | opcional | obligatorio | habilita `POST /api/tasks/ingest` |
| `WS_MAX_CONNECTION_SECONDS` | `0` | `280` | cierre ordenado antes del límite de la función |

- **Ingesta**: el cron de Vercel Hobby solo admite una ejecución al día, así
  que `.github/workflows/ingest.yml` llama cada 30 min a
  `POST /api/tasks/ingest` con `Authorization: Bearer <INGEST_TOKEN>`. Ejecuta
  `run_ingest` (lo mismo que el job del scheduler: rellena huecos, procesa y
  publica `observation`). Sin token configurado el endpoint responde **503**
  (nunca queda abierto); token ausente o incorrecto → **401**
  (`secrets.compare_digest`); fallo de la ingesta → **502**.
- **Reentrenamiento**: no se expone por HTTP (≈5 min, más que la duración
  máxima de la función). Se lanza a mano (`run_retraining`) o desde Docker.
- **Migraciones**: nunca por petición. Se aplican a mano contra Neon con la
  URL *directa* (sin `-pooler`): `alembic upgrade head`.
- `DATABASE_URL` acepta `postgres://`/`postgresql://` (formato de Neon) y se
  normaliza a `postgresql+psycopg://`.

### 9.4 Empaquetado para Vercel

- **Entrypoint**: `index.py` en la raíz exporta `app` (de `backend.app.main`).
- **Dependencias**: `requirements.txt` de la raíz = **runtime de la API**
  (también lo usa la imagen Docker). `backend/requirements.txt` lo incluye y
  añade lo exclusivo de los pipelines offline (`pyarrow`, `meteostat`).
  Medido en `python:3.12-slim` sin *bytecode*: **357 MB** de site-packages
  para la API frente a **484 MB** con todo, con un límite de 500 MB por
  función; más ~11 MB de artefactos `.joblib`.
- **`vercel.json`**: `maxDuration: 300`; `includeFiles` fuerza los `.joblib`
  y `backend/ml/config`; `excludeFiles` deja fuera `data/`, `docs/`,
  `frontend/`, `backend/tests/`, notebooks, `alembic/` y el archivo de
  modelos antiguos.
- `.python-version` fija Python 3.12 (el mismo que la imagen Docker).

### 9.5 Pruebas

`backend/tests/test_realtime.py`: bus en memoria, WebSocket (evento publicado,
`ping`, cierre 1011/1012, baja de la suscripción), autenticación de
`/tasks/ingest` (503/401/200/502, ingesta simulada), lifespan con el modo
serverless y configuración. Las de Redis llevan el marcador `redis`
(`docker compose up -d redis`). **85 pruebas** en total en verde.

---

## Q-gate fase 6 → fase 7

✅ API documentada con OpenAPI/Swagger · ingesta en tiempo real funcionando
(backfill + job cada 30 min, idempotente) · SSE operativo · comparación
histórica con percentiles condicionados · secretos solo en backend · CORS,
validación, manejo de errores y logging · Docker Compose.

**Siguiente (fase 7):** integración con la API de Claude — enviar el resumen
estructurado de `/insights` y devolver la interpretación en lenguaje natural,
con las reglas estrictas de la fase 1 (§18).
