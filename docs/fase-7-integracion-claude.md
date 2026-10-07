# FASE 7 — Integración con Claude (IA generativa)

**Ubicación:** Madrid-Barajas · **Fecha:** 2026-09-08
**Modelo:** `claude-haiku-4-5` (configurable, `ANTHROPIC_MODEL`)

---

## 1. El papel de Claude

Claude **no es el modelo de Machine Learning**. Es una **capa de interpretación**:
Python calcula todo (estadísticas, percentiles, régimen, anomalía, predicción,
métricas), lo empaqueta en un JSON acotado, y Claude lo traduce a un texto en
lenguaje natural con secciones fijas.

```
weather_observations + artefactos ML
   -> Python: build_ai_summary()  (resumen estructurado, ~20 campos numéricos)
   -> Claude (Anthropic API)      (system prompt estricto)
   -> texto en 7 secciones
   -> tabla ai_analyses           (con tokens consumidos)
```

**Nunca** se envían los registros históricos: solo el resumen. Un `GET /insights`
consume del orden de **850 tokens de entrada + 250 de salida ≈ 0,004 USD**.

## 2. El resumen estructurado (entrada a Claude)

`build_ai_summary()` (`backend/app/services/dashboard.py`) produce un JSON con la
forma de la fase 1, §18.3. Ejemplo real:

```json
{
  "location": "Madrid-Barajas",
  "period": "current",
  "current": {"temperature_c": 21.7, "humidity_pct": 51.0, "pressure_hpa": 945.9,
              "wind_kmh": 3.3, "precipitation_mm": 0.0, "cloud_cover_pct": 20.0},
  "historical_context": {
    "reference": "mismo mes ±1 y misma hora ±1, histórico completo",
    "temperature_percentile": 79.7, "humidity_percentile": 40.0,
    "pressure_percentile": 47.6, "normal_temp_range_c": [10.8, 23.5]
  },
  "regime": {"cluster_id": 2, "label": "frío, húmedo", "distance_to_centroid": 2.35},
  "anomaly": {"is_anomaly": false, "score": -0.403,
              "detector": "isolation_forest_on_residuals",
              "note": "valor inusual respecto al histórico; no implica fenómeno extremo"},
  "prediction": {"horizon_hours": 6, "predicted_temperature_c": 25.4,
                 "interval_c": [22.7, 27.9], "model_type": "hist_gradient_boosting",
                 "model_rmse_c": 1.62, "beats_persistence_baseline": true},
  "correlations_last_30d": {"temperature_relative": -0.68, ...},
  "data_quality": {"observation_age_minutes": 8.0, "delayed": false}
}
```

## 3. Reglas del *system prompt* (fase 1, §18.4)

`backend/app/services/claude/prompts.py`. Claude debe:

1. usar **solo** los datos del JSON; no inventar valores;
2. distinguir **hechos** de **interpretación**;
3. **no** afirmar causalidad;
4. presentar la predicción **con su banda de incertidumbre**, nunca como certeza;
5. tratar una anomalía como *"valor inusual respecto al histórico"*, no como
   fenómeno extremo;
6. recordar que el percentil está **condicionado a la época y la hora**;
7. responder en español, 200–400 palabras, en 7 secciones fijas: *Resumen,
   Patrones, Comparación histórica, Anomalías, Predicción, Interpretación,
   Recomendaciones de observación*.

## 4. Caché y coste acotado

- `GET /insights` devuelve el análisis en caché si tiene menos de
  `AI_CACHE_MINUTES` (60 por defecto). `?force=true` lo ignora.
- La ingesta en tiempo real genera un análisis **solo si detecta una anomalía**
  y ha pasado al menos `AI_MIN_GAP_MINUTES` (30) desde el último.
- El dashboard (`GET /dashboard`) **nunca** llama a la API: muestra el último
  análisis persistido en `ai_analyses`.

Techo realista: unas pocas llamadas al día → **muy por debajo de 1 USD/mes**.

## 5. *Fallback* (robustez)

| Situación | Respuesta de `/insights` |
|---|---|
| Sin `ANTHROPIC_API_KEY` | `status: "no_api_key"` + el resumen estructurado (sin interpretación) |
| API caída / error / *rate limit* | `status: "unavailable"` + la última interpretación disponible (`stale: true`) |
| Respuesta rechazada por seguridad | tratada como *unavailable* |

El resto del sistema **nunca** depende de Claude para funcionar.

## 6. Persistencia

Tabla `ai_analyses` (fase 1, §11):

| columna | contenido |
|---|---|
| `generated_at` | cuándo se generó |
| `period` | p. ej. `current` |
| `input_summary` (JSONB) | el JSON exacto que se envió a Claude (trazabilidad) |
| `model` | modelo que respondió (p. ej. `claude-haiku-4-5`) |
| `output_text` | la interpretación |
| `input_tokens` / `output_tokens` | consumo real (para controlar el gasto) |

## 7. Seguridad

- `ANTHROPIC_API_KEY` **solo** en `.env` del backend. No aparece en ninguna
  respuesta ni llega al frontend.
- El SDK oficial `anthropic` (1.4.0) con reintentos automáticos y *timeout* de 45 s.

## 8. Código

```
backend/app/services/claude/
  prompts.py       SYSTEM_PROMPT + plantilla del mensaje de usuario
  client.py        call_claude() — SDK anthropic, manejo de errores, is_configured()
  interpreter.py   get_or_create_interpretation() (caché + fallback + persistencia)
                   maybe_generate_on_event() (generación automática solo en anomalía)
```

`GET /api/insights` (`backend/app/api/routes/insights.py`) es el punto de entrada.

## 9. Pruebas

`backend/tests/test_claude.py` (5, con doble de `call_claude` — **no** se llama a
la API real): *fallback* sin clave, generación + persistencia + caché,
*fallback* a análisis anterior si la API falla, generación automática solo en
anomalía.

---

## Cómo activarlo

```bash
# en .env del backend
ANTHROPIC_API_KEY=sk-ant-...
```

Reiniciar el backend y `GET /api/insights?force=true`.

---

## Q-gate fase 7 → fase 8

✅ Claude como capa de interpretación (no de ML) · resumen estructurado acotado ·
system prompt con las reglas de la fase 1 · caché y coste acotado · *fallback*
completo · persistencia en `ai_analyses` con tokens · clave solo en backend ·
pruebas con doble.

**Siguiente (fase 8):** frontend React + Tailwind — dashboard con las 8
visualizaciones, tarjeta "AI Insights", y SSE para actualización en vivo.
