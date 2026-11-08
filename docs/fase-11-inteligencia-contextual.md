# Fase 11 — Inteligencia meteorológica contextual y explicable

> Evolución del producto: de «un panel con predicciones» a una **plataforma de inteligencia
> meteorológica predictiva, contextual y explicable**. Se añaden cinco módulos que responden
> preguntas concretas y siempre dicen con qué datos, cuántos casos y con qué límites lo hacen.
> Metodología: CRISP-ML(Q). Fecha de la verificación descrita aquí: **8 nov 2026**.

## 1. Los cinco módulos

| # | Pregunta que responde | Backend | Pestaña |
|---|---|---|---|
| F1 | ¿Qué tan normal es el tiempo de hoy? | `services/analysis/normality.py` | Análisis histórico |
| F2 | ¿En qué patrón (régimen) está la ciudad? | `services/analysis/regimes.py` | Análisis histórico |
| F3 | ¿Qué tan bien predice Weather Intelligence? | `services/evaluation/forecast_log.py` | Inteligencia predictiva |
| F4 | ¿Qué tan confiable es este pronóstico? | `services/evaluation/uncertainty.py` | Inteligencia predictiva |
| F5 | Pregúntale a Weather Intelligence | `services/analyst/` | Analista IA |

Principios transversales (aplican a los cinco):

- **Por ciudad.** Todo se calcula con los datos y modelos de la ciudad elegida (`?location=<slug>`).
  Ningún dato cruza de una ciudad a otra.
- **Honestidad ante todo.** Si no hay datos suficientes, el módulo lo declara
  (`status`, `notes`, estados «preliminar», «recopilando», «no disponible») en vez de
  inventar una conclusión. No existe ningún dato de relleno ni historial simulado.
- **Sin fuga de futuro.** Toda referencia estadística usa solo observaciones **anteriores** a la
  observación que se evalúa; las horas futuras que el *backfill* guarda con valores de
  pronóstico nunca se tratan como observaciones (ver §6.1).
- **Asociación ≠ causa.** Los textos distinguen *dato del sistema*, *asociación estadística* e
  *hipótesis general*; ninguna cifra se presenta como causa ni como probabilidad individual.

## 2. Fuentes de datos y sus límites

- **Histórico y «observado»:** Open-Meteo Historical Weather (reanálisis ERA5) y, en tiempo
  real, Open-Meteo Forecast API (`current`). **No son lecturas de estación**: el «observado» es
  un producto de reanálisis/modelo. Para Madrid-Barajas hubo contraste con Meteostat en la
  fase 2 (la velocidad del viento es la variable de peor acuerdo, r = 0,72); **las cinco
  ciudades colombianas no tienen contraste independiente con estaciones**.
- **Cobertura:** Madrid-Barajas desde 2000 (234 851 obs.) y cada ciudad colombiana desde 2010
  (147 765 obs.) hasta 8 nov 2026; 973 676 observaciones horarias en total.
- **Pronóstico de referencia:** el pronóstico archivado de Open-Meteo para la misma hora
  objetivo, guardado en `forecast_log` cuando se emite el pronóstico propio. Solo existe desde
  que se empezó a registrar (8 nov 2026).
- **Licencias:** CC BY 4.0 (Open-Meteo, Meteostat). La clave de Anthropic vive solo en el backend.

## 3. Métodos estadísticos

### F1 — Normalidad histórica

- **Referencia:** observaciones de la misma ciudad, mismo mes y misma franja horaria **local**
  (la hora local de la ciudad, no UTC), estrictamente anteriores a la observación actual.
- **Ventanas ampliables** (`LADDER`): mes/hora ±1 h (representatividad *alta*) → mes ±1 →
  mes ±1 y ±3 h (*media*) → mes ±2 → todo el año ±3 h (*baja*). Se usa la primera con al menos
  `MIN_SAMPLE = 300` observaciones; si ninguna llega, se declara «sin referencia suficiente».
- **Percentil** = % de la referencia con valor ≤ valor actual; rangos habituales p10–p90 y p25–p75;
  extremo univariable fuera de p2,5–p97,5 (solo el extremo alto para viento y lluvia).
- **Precipitación:** percentil solo entre horas con lluvia (≥ 0,1 mm, resolución de Open-Meteo) y
  con al menos 30 de ellas; si no, se informa la fracción de horas húmedas.
- **Anomalía multivariante:** el Isolation Forest entrenado en la fase 4 (por ciudad) puntúa la
  observación; se muestran las variables que más contribuyen. Rareza estadística **no** es peligro.
- **Caché** con TTL de 6 h (máx. 24 entradas) para no recalcular.

### F2 — Regímenes meteorológicos

- K-Means por ciudad (Bogotá y Madrid k = 3; Medellín, Cali, Barranquilla y Cartagena k = 2).
- **Etiquetas derivadas de estadísticos reales** del perfil de cada régimen (terciles de
  temperatura, humedad, nubosidad, viento y fracción de horas con lluvia), no escritas a mano.
- Las distancias al centroide y al segundo régimen se calculan **en el mismo espacio escalado**
  con el que se ajustó el modelo; un cociente d1/d2 cercano a 1 indica frontera difusa.
- **Transiciones** entre regímenes y **permanencia** (duración típica de los episodios) solo se
  muestran con soporte suficiente (≥ 30 cambios desde el origen; ≥ 10 episodios). Son
  frecuencias históricas, **no probabilidades de pronóstico**.
- Línea de tiempo de 7 días con las observaciones ya ocurridas. El perfil se calcula una vez y se
  guarda en `cluster_models.profile` (JSONB).

### F3 — Evaluación de pronósticos reales

- **Registro persistente** (`forecast_log`, tabla *append-only* con único
  `(ubicación, horizonte, hora base)`): cada ingesta registra los 5 horizontes (1, 3, 6, 12, 24 h)
  con versión del modelo, intervalo y referencia de Open-Meteo. Reintentar no duplica.
- **Verificación periódica e idempotente:** cuando llega la hora objetivo + `GRACE = 90 min`, se
  compara con la observación real. Sin observación utilizable pasados 14 días → `expired`.
  Una fila del tiempo real solo cuenta si se ingirió **después** de la hora objetivo.
- **Métricas por horizonte:** MAE, RMSE, sesgo (predicho − observado) y *n*, comparadas con la
  **persistencia** (skill = 1 − RMSE_modelo / RMSE_persistencia) y con el pronóstico de
  Open-Meteo **sobre las mismas filas**. Selector de 7 / 30 / 90 días.
- Se muestran por separado las pruebas **offline** (periodo de prueba histórico del
  entrenamiento) y los pronósticos **reales verificados**; nunca se mezclan.
- Con menos de 30 casos se avisa «muestra pequeña»; sin casos, «el registro acaba de empezar».

### F4 — Confiabilidad del intervalo

- **Nivel nominal 90 %.** La **cobertura empírica** es la frecuencia con que el valor real cayó
  dentro del intervalo. **No es la probabilidad de que un pronóstico concreto acierte.**
- **Calibración conformal móvil:** con ≥ 100 residuales verificados del mismo modelo (ventana de
  los 300 más recientes, ordenados en el tiempo) el intervalo se recalibra
  (`interval_method = conformal_rolling`). Antes se usa el de validación
  (`validation_residual_quantiles`).
- **Etiqueta** (alta / media / baja / preliminar / no evaluable) a partir de métricas verificadas:
  cobertura a menos de 5 puntos del nominal y sin deterioro → *alta*; a menos de 10 → *media*;
  peor → *baja*. **Deterioro:** MAE de los últimos 7 días 25 % por encima del de los 23 previos
  (o 50 % sobre el periodo de prueba). Sin 30 casos reales → *preliminar* (solo se conoce la
  cobertura offline). Los criterios exactos viajan en la respuesta (`criteria`).

### F5 — Analista

1. `build_context` reúne, **solo desde el backend**, el estado de la ciudad: observación,
   normalidad, régimen, pronóstico con intervalos, evaluación, modelos y la lista de lo que **no
   está disponible** (caché de 60 s).
2. **Respaldo determinista siempre disponible:** plantillas escritas con los datos reales. Sin
   clave de Claude el módulo funciona igual y lo declara («análisis determinista con datos reales»).
3. **Con Claude (opcional):** recibe únicamente el contexto estructurado y la pregunta como dato
   delimitado en `<pregunta>`. **Nunca recibe credenciales, JWT ni claves** (prueba automática).
   La respuesta debe separar `DATOS` / `ASOCIACION` / `HIPOTESIS`.
4. **Verificación numérica (`guard`):** si el texto de la IA contiene una cifra que no está en el
   contexto (tolerancia de redondeo), se **descarta** y se devuelve la respuesta determinista.
5. **Límites:** timeout de 25 s, 20 llamadas de IA y 120 preguntas por usuario y hora (429 con
   `Retry-After`), pregunta saneada a ≤ 300 caracteres, caché de 10 min para las preguntas
   sugeridas. Una pregunta que no encaja con ninguna intención recibe un `notice` explícito.

## 4. Endpoints nuevos

Todos requieren sesión (JWT) y aceptan `?location=<slug>`.

| Método y ruta | Parámetros | Devuelve |
|---|---|---|
| `GET /api/analysis/normality` | — | percentiles por variable, referencia (periodo, muestra, paso de ventana), anomalía, casos similares, `notes` |
| `GET /api/analysis/regimes` | — | perfiles, régimen actual, línea de tiempo 7 d, episodios, transiciones |
| `GET /api/evaluation/summary` | `days` 1–365 (def. 30) | métricas por horizonte, series diarias, modelos offline, `notes` |
| `GET /api/evaluation/pairs` | `days`, `horizon_hours`, `limit` ≤ 2000 | pares predicho/observado con intervalo |
| `GET /api/forecast/reliability` | — | por horizonte: pronóstico, intervalo, cobertura real/offline, etiqueta, `criteria` |
| `GET /api/analyst/questions` | — | las 7 preguntas sugeridas |
| `POST /api/analyst/ask` | cuerpo `{question?, intent?, detail}` | respuesta con `sections`, `evidence`, `limitations`, `availability`, `mode`, `fallback_reason`, `notice` |

Evento de tiempo real nuevo por WebSocket: `forecasts_verified` (`{location, n_verified}`).

## 5. Base de datos y migraciones

Solo cambios **aditivos** (sin borrar ni alterar datos existentes), con Alembic:

- `c3a9f1e5b8d2` — tabla `forecast_log` (índice único idempotente y índices de consulta) y
  columna `cluster_models.profile` (JSONB).
- `d5c8e2f1a7b3` — índice `ix_cluster_assignments_observation_id`. **Hallazgo:** al borrar
  observaciones, la cascada de la clave foránea hacía recorridos secuenciales completos sobre
  `cluster_assignments` (borrados de minutos); el índice los vuelve inmediatos.

`alembic check` → «No new upgrade operations detected».

## 6. Evidencia CRISP-ML(Q): defectos encontrados y corregidos

### 6.1 Fuga de futuro por el *backfill*
El *backfill* guarda la **hora siguiente** con valores de pronóstico (corte: ahora + 90 min). Eso
hacía que «la observación actual» pudiera ser una hora futura. Corrección: `latest_observation`
filtra `observed_at <= now()`, el módulo de regímenes descarta horas futuras y la verificación
solo acepta una fila de tiempo real ingerida **después** de su hora objetivo. Prueba incluida.

### 6.2 Modelo equivocado en la evaluación
Con un reentrenamiento posterior **no promovido**, el panel mostraba ese aspirante en vez del
campeón activo (y Madrid aparecía sin RMSE de prueba). Corrección: orden
`is_active desc, trained_at desc, id desc`. Prueba incluida.

### 6.3 Cobertura de los intervalos por debajo de la nominal (hallazgo, no corregido)
Evaluación offline en el tramo de prueba (`python -m backend.ml.modeling.evaluate_intervals`),
nivel nominal 90 %, cobertura por horizonte (1–24 h):

| Ciudad | Cobertura de prueba |
|---|---|
| Madrid-Barajas | 89,8 – 92,0 % |
| Bogotá | 90,5 – 91,9 % |
| Cartagena | 87,6 – 89,3 % |
| Medellín | 86,9 – 89,1 % |
| Cali | 85,3 – 90,2 % |
| Barranquilla | 84,8 – 86,7 % |

Medellín, Cali y Barranquilla **cubren menos de lo prometido** en la mayoría de horizontes. Por eso
F4 existe: etiqueta la confiabilidad con la cobertura medida y recalibra con pronósticos reales
verificados en cuanto hay datos suficientes. Es una evaluación **histórica**, no una garantía futura.

### 6.4 Otros
Placeholder de hora futura mostrado como «actual» (ver 6.1); `rowcount` incorrecto con
`ON CONFLICT DO NOTHING` (se usa `RETURNING`); contenido horizontal desbordado en móvil (panel
predictivo y cabecera a 320 px); título de la página fijo en «Madrid-Barajas».

## 7. Estado de los datos de evaluación real (honestidad)

El registro de pronósticos empezó el **8 nov 2026**: a fecha de esta verificación hay pronósticos
registrados pero **ninguno verificado todavía** (los horizontes aún no han llegado). Por eso el panel
muestra «El registro acaba de empezar» y la confiabilidad figura como **preliminar**, apoyada solo en
la cobertura offline. Las métricas reales (MAE/RMSE/sesgo, comparación con persistencia y con
Open-Meteo, cobertura real) aparecerán solas conforme se verifiquen. **No hay ningún valor de
relleno.**

## 8. Variables de entorno nuevas

Todas opcionales (valores por defecto entre paréntesis); ver `backend/app/core/config.py`:

| Variable | Significado |
|---|---|
| `ANTHROPIC_API_KEY` | Activa la redacción con Claude (sin ella, respuestas deterministas) |
| `ANALYST_TIMEOUT_SECONDS` (25) | Tiempo máximo de la llamada a Claude |
| `ANALYST_AI_PER_HOUR` (20) | Llamadas a Claude por usuario y hora |
| `ANALYST_ASKS_PER_HOUR` (120) | Preguntas totales por usuario y hora |
| `ANALYST_CACHE_MINUTES` (10) | Caché de respuestas de IA a preguntas sugeridas |

Redis sigue siendo opcional (bus de eventos en memoria si `REDIS_URL` está vacío).

## 9. Cómo ejecutarlo

```bash
docker compose up -d --build            # db + redis + backend (migraciones al arrancar) + frontend
# Añadir una ciudad del catálogo (descarga ERA5, entrena y publica; ~10-15 min):
PYTHONPATH=. backend/.venv/Scripts/python -m backend.ml.add_location --slug bogota
# Evaluar offline la cobertura de los intervalos de las ciudades entrenadas:
PYTHONPATH=. backend/.venv/Scripts/python -m backend.ml.modeling.evaluate_intervals --all
```

La app queda en `http://localhost:5173`. Tras registrarte, las cuatro pestañas
(Vista general, Análisis histórico, Inteligencia predictiva, Analista IA) funcionan para cada ciudad
del selector.

## 10. Pruebas: qué se ejecutó y qué no

- **Backend:** 214 pruebas pasan y 3 quedan deseleccionadas (`-m "not network and not redis"`,
  70 s); ruff sin avisos; `alembic check` sin cambios pendientes. Cubren: percentiles y ventanas sin fuga de futuro y con muestra insuficiente;
  regímenes (etiquetas, distancias, transiciones, línea de tiempo); registro idempotente,
  verificación, persistencia y referencia; conformal con orden temporal; etiquetas de confiabilidad;
  analista (sin clave, con IA simulada, cifras no respaldadas, inyección de prompt, límites de
  uso, ninguna credencial enviada).
- **No ejecutadas:** 1 prueba `network` (requiere internet) y 2 `redis` (requieren Redis en marcha).
- **La ruta de IA del analista solo se probó con un cliente simulado**: no se hizo ninguna llamada
  real a Claude porque no hay clave configurada en este entorno.
- **Extremo a extremo con Docker Compose** (imágenes reconstruidas, acceso por el nginx del
  frontend en `:5173`): con una cuenta de prueba local, `normality`, `regimes`, `evaluation/summary`,
  `forecast/reliability` y `analyst/ask` respondieron 200 en las seis ciudades (el analista en modo
  determinista); sin token responde 401 y con una ciudad inexistente 404. Tiempo de respuesta
  habitual < 1 s; la primera consulta de regímenes de una ciudad tardó ~3 s porque calcula y guarda
  su perfil.
- **Frontend:** no existe un ejecutor de pruebas unitarias; se verificó con `tsc` estricto sin
  errores y manualmente en el navegador: pestañas, cambio de ciudad (el estado se limpia y se vuelve
  a pedir), estados «recopilando/preliminar», analista con preguntas sugeridas y libres, ausencia de
  errores en consola y de desbordamiento horizontal a 375 y 320 px.
