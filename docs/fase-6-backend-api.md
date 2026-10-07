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
    routes/            health, weather, analytics, predictions, dashboard, insights, stream
  schemas/models.py    modelos Pydantic de respuesta
  services/
    ingestion/         WeatherProvider (adapter) + open_meteo + realtime
    historical/        comparison — percentiles condicionados, régimen, anomalía
    prediction/        TemperaturePredictor (carga artefactos, predice)
    artifacts.py       carga perezosa y cacheada de los .joblib
    events.py          bus de eventos en proceso para SSE
    dashboard.py       ensamblado del /dashboard y del resumen para Claude
  scheduler/jobs.py    APScheduler: job de ingesta cada 30 min
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

---

## Q-gate fase 6 → fase 7

✅ API documentada con OpenAPI/Swagger · ingesta en tiempo real funcionando
(backfill + job cada 30 min, idempotente) · SSE operativo · comparación
histórica con percentiles condicionados · secretos solo en backend · CORS,
validación, manejo de errores y logging · Docker Compose.

**Siguiente (fase 7):** integración con la API de Claude — enviar el resumen
estructurado de `/insights` y devolver la interpretación en lenguaje natural,
con las reglas estrictas de la fase 1 (§18).
