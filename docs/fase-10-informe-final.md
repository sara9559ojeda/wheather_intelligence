# Weather Intelligence
## Informe final — plataforma de minería de datos, Machine Learning e IA generativa para el análisis meteorológico de Madrid-Barajas

**Autora:** Sara Ojeda · **Metodología:** CRISP-ML(Q) · **Fecha:** septiembre de 2026
**Repositorio:** proyecto local (`weather-intelligence`) · **Ubicación analizada:** Madrid-Barajas (40.4936, −3.5668)

> Este documento consolida las diez fases del proyecto (docs/fase-1 a fase-9)
> en un único informe apto para defensa. Cada afirmación cuantitativa remite
> al documento de fase que la sustenta; los números no se han re-derivado
> aquí, se han **copiado** de las ejecuciones reales registradas en esos
> documentos y en la propia base de datos del sistema.

---

## 0. Resumen ejecutivo

Weather Intelligence es un sistema completo que integra **adquisición de
datos, minería de datos, Machine Learning e IA generativa** para el análisis
meteorológico de una ubicación real (Madrid-Barajas), siguiendo la
metodología **CRISP-ML(Q)** de principio a fin.

El sistema:

1. Entrena con un **histórico real de 233.880 observaciones horarias**
   (2000–2026, reanálisis ERA5 vía Open-Meteo, licencia CC BY 4.0), contrastado
   contra una estación meteorológica real (Meteostat).
2. Descubre **3 regímenes meteorológicos** mediante K-Means (silhouette 0,247,
   validado por el método del codo) y detecta **anomalías** (2,07 % de las
   horas) con Isolation Forest sobre residuales des-estacionalizados.
3. Predice la temperatura a **5 horizontes (1–24 h)** con un modelo campeón
   (*Histogram Gradient Boosting*) que **supera a la persistencia en todos los
   horizontes** (skill +16 % a +77 %) y evaluado con partición **cronológica**
   para evitar fuga de datos.
4. Se mantiene vivo: ingiere datos en tiempo real cada 30 minutos, compara
   cada observación nueva contra su histórico, y **se reentrena solo cuando
   corresponde**, con una regla de promoción que impide degradar el sistema
   en producción — verificada con una ejecución real.
5. Complementa los resultados numéricos con una interpretación en lenguaje
   natural generada por Claude (Anthropic), siempre a partir de un resumen
   estructurado calculado por Python — nunca al revés.
6. Todo lo anterior se sirve mediante una **API REST documentada** (FastAPI)
   y un **dashboard web responsive** (React + Tailwind + Recharts) con
   actualización en vivo.

El sistema, la base de datos y los 63 tests automáticos están en
funcionamiento y verificados con datos reales al cierre de este informe.

---

## 1. Introducción

### 1.1 Planteamiento del problema

Existe una enorme cantidad de datos meteorológicos públicos, tanto
históricos como en tiempo real, de calidad alta y acceso libre. Sin embargo,
entre "el dato crudo" y "una conclusión útil para una persona" hay un
trabajo técnico considerable: adquirir los datos, almacenarlos de forma
consistente, limpiarlos, transformarlos, aplicar minería de datos y Machine
Learning, **validar los modelos correctamente** (algo especialmente delicado
en series temporales, donde es fácil cometer fuga de datos sin darse
cuenta) y, finalmente, comunicar los resultados de forma comprensible sin
exagerar su certeza.

No existía, al iniciar este proyecto, un sistema integrado y reproducible
que combinara una fuente histórica grande con una fuente en tiempo real bajo
un mismo esquema de datos, ejecutara un pipeline reproducible de principio a
fin, validara los modelos respetando su naturaleza temporal, contextualizara
cada observación nueva frente al histórico, y tradujera esos resultados a
lenguaje natural de forma responsable.

**Pregunta central del proyecto:** *¿es posible construir un sistema que, a
partir de datos meteorológicos históricos y actuales, descubra patrones,
detecte condiciones inusuales y prediga la temperatura futura con un error
acotado y medible, manteniéndose actualizado con el tiempo y explicando sus
resultados de forma responsable?*

### 1.2 Justificación

**Académica.** El proyecto recorre de forma completa y verificable las
etapas de un proceso real de minería de datos y ML bajo CRISP-ML(Q):
adquisición, ETL, *data understanding*, *data preparation*, EDA, minería de
datos no supervisada (clustering y anomalías), modelado supervisado con
comparación y selección de modelos, validación temporal correcta,
mantenimiento (reentrenamiento) e integración de IA generativa como capa de
interpretación — no como sustituto del análisis.

**Técnica.** Integra un *stack* profesional realista: API REST documentada,
base de datos relacional normalizada (9 tablas, claves foráneas e índices),
contenedores Docker, frontend moderno, separación estricta de
responsabilidades, manejo de secretos, *logging*, pruebas automatizadas y
reproducibilidad (semillas fijas, versiones ancladas, metadatos con
`sha256`).

**Práctica.** El resultado es utilizable: un dashboard que muestra
condiciones actuales, cómo se comparan con lo normal para esa época del año,
si hay algo inusual, y qué predice el modelo — con una explicación en
lenguaje natural y su incertidumbre siempre visible. El patrón (histórico +
tiempo real + minería + ML + capa de interpretación) es transferible a otros
dominios con series temporales (energía, calidad del aire, demanda).

### 1.3 Objetivo general

Construir un sistema inteligente y reproducible capaz de analizar datos
meteorológicos históricos y actuales para **descubrir patrones**, **detectar
anomalías** y **generar predicciones de temperatura** mediante técnicas de
minería de datos y aprendizaje automático, complementando la presentación de
los resultados con una capa de interpretación basada en inteligencia
artificial generativa, bajo la metodología CRISP-ML(Q).

### 1.4 Objetivos específicos

1. Adquirir e integrar dos fuentes de datos meteorológicos (histórico ≥
   50.000 registros reales + API en tiempo real) bajo un esquema común en
   PostgreSQL, mediante ETL reproducible. — **Fases 2, 6**
2. Caracterizar y documentar los datos (*data understanding*): volumen,
   variables, faltantes, duplicados, rangos, distribuciones y estructura
   temporal; producir un diccionario de datos. — **Fase 2**
3. Construir un pipeline de preparación reproducible con *feature
   engineering* justificado (temporales, cíclicas, retardos, medias
   móviles). — **Fase 3**
4. Realizar un EDA completo (univariado, bivariado, temporal, correlaciones)
   con interpretación escrita. — **Fase 3**
5. Aplicar clustering K-Means determinando *k* con Elbow + Silhouette, e
   interpretar cada régimen desde los propios datos. — **Fase 4**
6. Implementar detección de anomalías con Isolation Forest sobre residuales
   des-estacionalizados. — **Fase 4**
7. Entrenar y comparar modelos de regresión (baselines + Ridge + Random
   Forest + Gradient Boosting) por horizonte de predicción. — **Fase 5**
8. Validar con partición cronológica (70/15/15), seleccionar el modelo por
   regla escrita y evaluar en test una sola vez. — **Fase 5**
9. Contextualizar cada observación nueva frente al histórico (percentiles
   condicionados, régimen, anomalía, predicción con incertidumbre). —
   **Fases 6, 9**
10. Integrar la API de Claude como capa de interpretación, con resumen
    acotado y reglas estrictas. — **Fase 7**
11. Exponer una API backend documentada y un dashboard responsive con
    actualización en vivo. — **Fases 6, 8**
12. Diseñar una estrategia de reentrenamiento con criterio de promoción
    explícito y verificarla con una ejecución real. — **Fase 9**
13. Empaquetar el sistema con Docker Compose, con seguridad, *logging* y
    pruebas automatizadas. — **Fases 6, 8, 9**

---

## 2. Preguntas de investigación

Formuladas en la Fase 1 (§6) y respondidas con evidencia recogida durante el
proyecto.

### P1 — ¿Qué patrones meteorológicos aparecen históricamente?

**Dos ciclos dominan la serie:** el diario (mínimo hacia las 06:00 UTC, 9,4 °C
de media; máximo hacia las 15:00 UTC, 20,0 °C) y el anual (enero 4,9 °C →
julio 25,9 °C). La minería de datos, sin que se le indicara la hora ni el
mes, encontró **3 regímenes** ("cálido-seco-soleado" 37 % de las horas,
"frío-húmedo-tormentoso" 15 %, "frío-húmedo-tranquilo" 48 %) cuya frecuencia
por mes y hora reproduce exactamente esos ciclos (Fase 4).

### P2 — ¿Existe relación entre temperatura y humedad?

Sí, **negativa y fuerte**: Pearson = **−0,795** (Fase 3, EDA). El aire más
cálido admite más vapor de agua, así que a igual humedad absoluta la
relativa baja. Es la variable con mayor correlación (en valor absoluto) con
la temperatura de todo el conjunto.

### P3 — ¿Qué variables tienen mayor relación con la temperatura?

Por correlación lineal: humedad relativa (−0,795), radiación solar (+0,551),
punto de rocío (+0,498). Por **importancia en el modelo predictivo**
(permutación, Fase 5): a 1 h vista domina el retardo de 1 h de la propia
temperatura (persistencia); a horizontes de 6–24 h ganan peso las variables
cíclicas (`hour_sin/cos`, `doy_cos`) y la media móvil de 24 h — el modelo
"aprende" el ciclo diario y anual igual que lo hizo el EDA.

### P4 — ¿Cuáles son las condiciones meteorológicas normales?

Se definieron de forma **condicionada a la época del año y la hora** (no
contra todo el año — sería engañoso comparar un mediodía de julio contra la
media anual). El sistema calcula, para cada observación nueva, el percentil
de temperatura/humedad/presión/viento dentro de la ventana histórica de
mes±1 y hora±1 (Fase 6, `comparison.py`). Ejemplo real verificado en
producción: 33,6 °C en septiembre a mediodía cayó en el **percentil 89**
(rango normal para esa época: 16,3–33,8 °C).

### P5 — ¿Qué condiciones pueden considerarse anómalas?

Se marcaron **4.838 de 233.880 horas (2,07 %)** con Isolation Forest sobre
**residuales des-estacionalizados** (valor observado − climatología de ese
mes×hora), no sobre los valores crudos — de lo contrario todo el verano
habría salido "anómalo" frente a la media anual. Las anomalías más extremas
combinan presión muy baja (≈920 hPa frente a los ≈946 hPa habituales) con
viento fuerte: el sistema, sin que nadie se lo dijera, señaló como más
anómala la borrasca real *Aline* (octubre de 2023) que afectó a la
península. Se comunica siempre con el matiz: una anomalía estadística es
*"valor inusual respecto al histórico"*, no necesariamente un fenómeno
extremo.

### P6 — ¿Podemos agrupar diferentes condiciones meteorológicas?

Sí — K-Means con *k* = 3, elegido con el método del codo **y** el silhouette
(ambos coinciden, Fase 4). Los 3 grupos son físicamente interpretables y sus
etiquetas se derivaron automáticamente de la desviación de cada centroide
respecto a la media global, no se escribieron a mano.

### P7 — ¿Podemos predecir la temperatura futura?

Sí, con error acotado y medido: RMSE de test de **0,72 °C a 1 h** hasta
**2,16 °C a 24 h** (Fase 5), con R² > 0,94 en los 5 horizontes. El modelo
campeón (Histogram Gradient Boosting) **supera a la persistencia en los 5
horizontes** — el requisito mínimo para que un modelo de ML se considere
útil frente al "seguirá igual".

### P8 — ¿Qué tan diferente es el clima actual respecto al histórico?

Se responde con el mismo mecanismo de P4: percentiles condicionados por
variable, mostrados en el dashboard con su rango normal. El sistema también
informa si la observación se clasifica como anómala (P5).

### P9 — ¿A qué patrón histórico se parece el clima actual?

Cada observación nueva se asigna al cluster K-Means más cercano (distancia
euclídea tras estandarizar) y se muestra su etiqueta y distancia al
centroide — verificado en producción: una tarde fría y húmeda de septiembre
se asignó al régimen "frío, húmedo" con distancia 2,35.

### P10 (añadida durante el proyecto) — ¿El sistema puede mejorar solo con el tiempo, sin empeorar?

Se diseñó y **se ejecutó de verdad** un mecanismo de reentrenamiento con
criterio de promoción (Fase 9): con 210 observaciones nuevas acumuladas
(~0,1 % del dataset), el sistema entrenó 5 modelos retadores, los comparó
contra los campeones activos **sobre el mismo conjunto de test**, y
**conservó correctamente los 5 campeones** porque la mejora no superaba el
2 % exigido — demostrando que el mecanismo protege el sistema de
sustituciones injustificadas, tal como se diseñó.

---

## 3. Metodología: CRISP-ML(Q)

El proyecto siguió explícitamente las 6 fases de CRISP-ML(Q), cada una con
su propia puerta de calidad (*Q-gate*) antes de continuar a la siguiente.

| Fase CRISP-ML(Q) | Correspondencia en el proyecto | Q-gate |
|---|---|---|
| **1. Business & Data Understanding** | Fase 1 (diseño) + Fase 2 (perfilado de datos) | Diccionario de datos completo; dataset ≥ 50.000 filas cargado y perfilado |
| **2. Data Preparation** | Fase 3 | Pipeline reproducible; 38 *features* justificadas; test automático anti-fuga en verde |
| **3. Modeling** | Fases 4 y 5 | *k* elegido con Elbow+Silhouette; ≥ 3 modelos + 2 *baselines* comparados por horizonte |
| **4. Evaluation** | Fase 5 (test) + Fase 9 (reentrenamiento) | Modelo supera a la persistencia; regla de promoción escrita y verificada |
| **5. Deployment** | Fases 6, 7, 8 | API documentada; dashboard funcionando; Docker Compose |
| **6. Monitoring & Maintenance** | Fase 9 | Ingesta en tiempo real; disparador de reentrenamiento; trazabilidad completa |

Cada una de las 9 fases de trabajo produjo su propio documento técnico
(`docs/fase-N-*.md`), con resultados generados por ejecución real de código
contra datos reales — no hay cifras estimadas o inventadas en ninguna parte
de este informe.

---

## 4. Arquitectura del sistema

```
Open-Meteo (ERA5, histórico)  +  Meteostat (contraste)  +  Open-Meteo Forecast (tiempo real)
                    │
                    ▼
        ETL (validación, limpieza, upsert idempotente)
                    │
                    ▼
        PostgreSQL 16  (9 tablas, ver §5)
                    │
        ┌───────────┼──────────────────────────────┐
        ▼           ▼                              ▼
  Data Preparation  Minería de datos          Comparación histórica
  (features, split) (K-Means, Isolation Forest) (percentiles condicionados)
        │           │                              │
        ▼           ▼                              │
  Machine Learning (Ridge / RF / HistGB)            │
        │                                           │
        └──────────────┬────────────────────────────┘
                        ▼
              Resumen estructurado (Python)
                        │
                        ▼
               API de Claude (interpretación)
                        │
                        ▼
         Backend API (FastAPI) — OpenAPI/Swagger, SSE
                        │
                        ▼
      Frontend (React + Tailwind + Recharts) — dashboard en vivo
```

**Decisión de arquitectura:** monolito modular, no microservicios. Tres
contenedores en `docker-compose.yml` (`db`, `backend`, `frontend`) —
suficiente y más defendible que una arquitectura distribuida innecesaria
para el volumen de datos y tráfico del proyecto.

**Capas del backend** (`backend/app/` y `backend/ml/`):

- `ml/` — analítica *offline*: adquisición, preparación, minería, modelado,
  reentrenamiento. Ejecutable como módulos independientes (`python -m
  backend.ml....`).
- `app/services/features/` — construcción de *features*, **compartida**
  entre entrenamiento e inferencia en tiempo real (evita *train-serving
  skew*: el mismo código genera las variables en ambos casos).
- `app/services/{ingestion, historical, prediction, claude}/` — ingesta en
  tiempo real, comparación histórica, inferencia, interpretación con IA.
- `app/api/routes/` — 17 endpoints REST documentados con OpenAPI.
- `app/scheduler/` — dos trabajos periódicos (ingesta cada 30 min,
  comprobación de reentrenamiento cada 24 h), ambos robustos a fallos.

---

## 5. Fuentes de datos, ETL y base de datos

### 5.1 Fuentes (decisión justificada en Fase 1, verificada en Fase 2)

| Fuente | Uso | Registros | Licencia |
|---|---|---:|---|
| **Open-Meteo Historical (ERA5)** | Histórico de entrenamiento | 233.880 horas (2000-01-01 → 2026-09-05) | CC BY 4.0 |
| **Meteostat, estación 08221 (LEMD)** | Contraste de fidelidad | 228.281 horas, 99,2 % cobertura | CC BY 4.0 |
| **Open-Meteo Forecast** | Ingesta en tiempo real | ~30 min/registro desde el despliegue | CC BY 4.0 |

Se eligió **el mismo proveedor (Open-Meteo)** para histórico y tiempo real
para que las *features* de entrenamiento y de inferencia sean idénticas —
razón técnica de peso frente a mezclar proveedores.

**Contraste de fidelidad de ERA5** (Fase 2, el dato que justifica usar
reanálisis en vez de solo observación de estación): para la temperatura,
MAE = 1,31 °C y correlación = 0,985 frente a la estación real; para el
viento, la variable meteorológica más difícil, correlación = 0,72 —
limitación explícitamente documentada y tenida en cuenta en el diseño de
*features*.

### 5.2 ETL y esquema de base de datos

El ETL histórico (`backend/app/services/etl/`) valida contra rangos físicos
plausibles, tipa, y hace *upsert* idempotente en PostgreSQL — repetir la
carga no duplica nada gracias a `UNIQUE(location_id, observed_at, source)`.

**9 tablas** (Fase 1 §11, migradas con Alembic):

| Tabla | Propósito |
|---|---|
| `locations` | Catálogo de ubicaciones (Madrid-Barajas) |
| `weather_observations` | Observaciones históricas y en tiempo real (233.880 + ~210 y creciendo) |
| `model_runs` | Cada modelo entrenado (30 filas: 25 de la fase 5 + retadores de la fase 9), con métricas e hiperparámetros |
| `predictions` | Predicciones del campeón sobre el test (~175.000) y en producción |
| `cluster_models` / `cluster_assignments` | El modelo K-Means activo y la asignación de cada observación |
| `anomalies` | Puntuación de anomalía de cada observación |
| `ai_analyses` | Interpretaciones de Claude, con tokens consumidos |
| `retraining_runs` | Cada intento de reentrenamiento, se promocione o no |

---

## 6. Data Understanding y Data Preparation

**Data Understanding** (Fase 2): el histórico está **impecable** — 0 nulos,
0 duplicados, 0 huecos temporales en 233.880 filas (cobertura horaria
100 %). Solo 4 valores de presión ligeramente fuera del rango plausible
inicial (anticiclones reales, no errores). El contraste con Meteostat
(estación real) reveló, en cambio, datos "sucios" típicos de una fuente de
observación (31 valores de humedad > 100 %, columnas con más del 85 % de
huecos) — documentado como material de comparación, no usado para entrenar.

**Data Preparation** (Fase 3): pipeline reproducible de limpieza (0 cambios
necesarios en ERA5) + **38 *features* derivadas**, cada una justificada:
calendario y codificación cíclica (`hour_sin/cos`, `doy_sin/cos` — para que
el modelo entienda que la medianoche y las 23:59 están próximas), dirección
del viento como variable circular, retardos de 1 a 24 h, medias/desviaciones
móviles calculadas **hacia atrás únicamente** (nunca incluyen el instante
actual), tendencia barométrica, variables binarias.

**Partición cronológica**, no aleatoria — la decisión más importante contra
la fuga de datos en series temporales: 70 % train (2000–2018), 15 % valid
(2018–2022), 15 % test (2022–2026), con un **hueco de 24 h** entre tramos
igual al horizonte máximo de predicción, para que ningún *target* del final
de un tramo se apoye en datos del tramo siguiente. Verificado con tests
automáticos dedicados (`test_features.py`, `test_split.py`), incluida una
prueba que corrompe deliberadamente el futuro y comprueba que las *features*
del pasado no cambian.

---

## 7. Análisis Exploratorio de Datos (EDA)

Realizado sobre train + valid (2000–2022); el test se reservó sin tocar
hasta la evaluación final de la Fase 5.

- **Univariado:** la temperatura tiene media 14,5 °C y desviación 9,0 °C
  (ancha, por el ciclo anual); precipitación y radiación están muy sesgadas
  a cero.
- **Bivariado:** confirma P2 y P3 (arriba).
- **Temporal:** confirma el doble ciclo (P1). Sin señal por día de la semana
  — comprobación de que no hay artefactos de muestreo.
- **Correlaciones:** matriz completa calculada y presentada, con la
  advertencia explícita de que **correlación no implica causalidad** — por
  ejemplo, radiación y temperatura suben juntas porque ambas dependen de la
  posición solar, no porque una cause directamente a la otra en estos datos.

---

## 8. Minería de datos

### 8.1 Clustering (K-Means)

Variables: estado físico instantáneo (temperatura, humedad, presión,
viento, nubosidad, radiación, precipitación con `log1p`) — deliberadamente
**sin** variables de tiempo, para que los regímenes surjan del propio estado
del tiempo, no de la hora. *k* se determinó evaluando de 2 a 10 con
**Elbow** e **Silhouette**; ambos coinciden en *k* = 3 (silhouette 0,247).

| Cluster | % horas | Etiqueta (derivada de los datos) | Temp. media |
|---|---:|---|---:|
| 0 | 36,8 % | cálido, seco, soleado | 23,4 °C |
| 1 | 15,0 % | frío, húmedo, nublado, ventoso, con precipitación | 9,6 °C |
| 2 | 48,2 % | frío, húmedo | 9,1 °C |

### 8.2 Detección de anomalías (Isolation Forest)

Sobre **residuales des-estacionalizados** (corrección de diseño clave de
este proyecto: aplicarlo a valores crudos habría marcado el verano entero
como anómalo). 4.838 de 233.880 horas (2,07 %) marcadas; las más extremas
combinan presión muy baja y viento fuerte — coherentes con episodios de
borrasca reales. Comunicación siempre responsable: anomalía estadística ≠
fenómeno extremo garantizado.

---

## 9. Machine Learning: predicción de temperatura

**Comparación** (Fase 5) por horizonte: 2 *baselines* (persistencia,
climatología estacional) + 3 modelos (Ridge, Random Forest, **Histogram
Gradient Boosting** — sustituto moderno y ~50× más rápido de
`GradientBoostingRegressor`, mismo método, justificado en el código) sobre
**validación**; selección del campeón por regla escrita (menor RMSE de
validación); **una sola** evaluación en test.

### Resultado final (test, evaluado una vez)

| Horizonte | Modelo | MAE | RMSE | R² | Skill vs. persistencia |
|---|---|---:|---:|---:|---:|
| 1 h | HistGB | 0,49 | 0,73 | 0,994 | **+0,47** |
| 3 h | HistGB | 0,92 | 1,22 | 0,983 | **+0,66** |
| 6 h | HistGB | 1,21 | 1,56 | 0,972 | **+0,75** |
| 12 h | HistGB | 1,47 | 1,87 | 0,960 | **+0,77** |
| 24 h | HistGB | 1,69 | 2,16 | 0,946 | **+0,16** |

*Skill vs. persistencia* = 1 − RMSE_modelo / RMSE_persistencia; positivo en
los 5 horizontes ⇒ el modelo siempre aporta sobre "seguirá haciendo lo
mismo". **Hallazgo destacable:** el *skill* no crece monótonamente — es
máximo a 6–12 h y cae a 24 h, porque a 24 h la persistencia ya es un rival
razonable (compara la misma hora del día siguiente); a 12 h es un rival
pésimo (compara, por ejemplo, mediodía con medianoche). El modelo, en
cambio, mantiene un error bajo y estable en todo el rango.

**Incertidumbre:** cada predicción se acompaña de una banda `[p05, p95]` de
los residuales de validación — la predicción nunca se presenta como
certeza.

---

## 10. Evaluación final y validación temporal

El diseño de validación es, junto con la minería de datos, la parte más
defendible metodológicamente del proyecto:

- Partición **estrictamente cronológica** (§6), no barajada.
- `TimeSeriesSplit` (3 *folds*, ventana expansiva) como validación cruzada
  de robustez adicional sobre *train* — RMSE coherente entre *folds*.
- **Importancia de variables por permutación** (funciona con cualquier
  modelo, no solo con árboles): confirma que a corto plazo domina la
  persistencia y a largo plazo dominan las variables cíclicas — coherente
  con el EDA (§7) y con la intuición física.
- El conjunto de test se evaluó **una sola vez por horizonte**, y de nuevo
  **una sola vez más** durante el reentrenamiento real de la Fase 9 (sobre
  un test recalculado con datos más recientes, no el mismo).

---

## 11. Integración con IA generativa (Claude)

Claude **no es el modelo de Machine Learning** — es una capa de
interpretación en lenguaje natural sobre resultados que Python ya calculó.

```
Python calcula: percentiles condicionados, régimen, anomalía, predicción,
                métricas del modelo, correlaciones recientes
        │
        ▼
Resumen estructurado (~20 campos numéricos, JSON acotado)
        │
        ▼
Claude (claude-haiku-4-5) con system prompt estricto:
  usar solo los datos dados · distinguir hecho de interpretación ·
  no afirmar causalidad · predicción siempre con incertidumbre ·
  anomalía ≠ fenómeno extremo · 7 secciones fijas
        │
        ▼
Texto interpretativo persistido en ai_analyses (con tokens consumidos)
```

**Nunca** se envían registros históricos completos a la API — solo el
resumen. Coste estimado por llamada: ≈ 0,004 USD. Caché de 60 minutos y
generación automática solo ante una anomalía detectada (con un espaciado
mínimo), para acotar el gasto. **El sistema nunca depende de Claude para
funcionar**: sin clave configurada, `/insights` devuelve el resumen
estructurado sin interpretación; si la API falla, se sirve el último
análisis disponible marcado como desactualizado.

---

## 12. Sistema en producción: backend, frontend y tiempo real

**Backend** (FastAPI): 17 endpoints documentados con OpenAPI/Swagger
(`/docs`), CORS restringido, validación Pydantic en toda entrada, manejo de
errores centralizado, *logging* estructurado. **Ingesta en tiempo real**
cada 30 minutos: descarga las condiciones actuales de Open-Meteo, las
alinea a la hora, valida, guarda de forma idempotente y dispara el
procesamiento completo (predicción a 5 horizontes, anomalía, régimen,
comparación histórica) — verificado en vivo con datos reales de septiembre
de 2026 (33,6 °C, percentil 89 para la época, sin anomalía, régimen
"cálido-seco-soleado").

**Actualización al dashboard vía Server-Sent Events** (elegido frente a
*polling* y WebSockets: el dashboard solo recibe avisos, nunca envía nada —
patrón *"SSE notifica, REST trae los datos"*).

**Frontend** (React + TypeScript + Tailwind + Recharts + TanStack Query):
dashboard responsive con las 8 visualizaciones requeridas (temperatura y
humedad vs. tiempo, dispersión temperatura-humedad, matriz de correlación,
regímenes, anomalías, predicho vs. real, comparación histórico-actual),
estados de carga/error/vacío explícitos, accesibilidad (HTML semántico,
`:focus-visible`, contraste, nada transmitido solo por color). Verificado
funcionando en escritorio y en móvil (375×812) con datos reales en vivo.
Ninguna clave de API ni lógica de negocio en el navegador — el frontend
**solo pinta** lo que el backend le sirve.

**Empaquetado:** `docker-compose.yml` con 3 servicios (`db`, `backend`,
`frontend`); migraciones Alembic automáticas al arrancar el backend.

---

## 13. Reentrenamiento y mantenimiento

Diseñado para ser **ligero** (sin Airflow ni MLflow, como se pidió) pero
**seguro**: el scheduler comprueba cada 24 h si toca reentrenar (por volumen
≥ 1.000 observaciones nuevas, o por calendario ≥ 30 días), y si toca,
entrena un modelo "retador" por horizonte y lo compara contra el "campeón"
activo **sobre el mismo conjunto de test recién construido** — nunca
comparando contra una métrica antigua de un test distinto.

**Regla de promoción, escrita y determinista:** el retador sustituye al
campeón solo si mejora su RMSE ≥ 2 %, o si el campeón ha degradado más de un
20 % desde que se entrenó y el retador no es peor. Todo intento —se
promocione o no— queda registrado, y los artefactos sustituidos se archivan,
nunca se borran.

**Verificación real** (no simulada): con 210 observaciones nuevas
acumuladas, el sistema entrenó los 5 retadores, los comparó de forma justa y
**conservó correctamente los 5 campeones** (diferencias de RMSE de
0,001–0,003 °C, muy por debajo del 2 % exigido) — el resultado esperado con
tan poco volumen nuevo, y la prueba de que la regla protege al sistema en
producción en vez de sustituirlo por ruido.

---

## 14. Resultados y demostración

- **233.880** observaciones históricas + ingesta en tiempo real activa.
- **9** tablas en PostgreSQL, **17** endpoints REST documentados.
- **3** regímenes meteorológicos interpretables descubiertos por K-Means.
- **2,07 %** de horas marcadas como anómalas por Isolation Forest, con
  ejemplos verificablemente coherentes con eventos reales.
- **5** modelos de predicción de temperatura, todos superando a la
  persistencia, con RMSE de 0,73 a 2,16 °C.
- **~175.000** predicciones históricas persistidas + predicción en vivo
  funcionando.
- **1** ejecución de reentrenamiento real completada y auditada.
- **63 pruebas automatizadas** en verde, `ruff` limpio en todo el backend.
- Dashboard completo, responsive, con datos en vivo, verificado con
  capturas reales durante el desarrollo.

---

## 15. Conclusiones

1. **Es posible** construir, con herramientas de código abierto y sin
   infraestructura pesada, un sistema completo de minería de datos y ML
   meteorológico que se mantiene actualizado y se explica solo — la
   pregunta central del proyecto (§1.1) queda respondida afirmativamente y
   con evidencia reproducible.
2. La disciplina metodológica (CRISP-ML(Q), partición cronológica, *Q-gates*
   por fase) no fue un formalismo: **evitó errores concretos** que se habrían
   colado con un enfoque menos cuidadoso — por ejemplo, comparar contra la
   media anual en vez de contra la época del año (P4), o marcar el verano
   entero como anómalo (P5).
3. El modelo de Machine Learning aporta un valor medible y matizado: gana
   mucho a la persistencia a 6–12 h, y menos a 24 h — un resultado más
   honesto e interesante que un número único de "precisión".
4. Separar Machine Learning (que decide) de IA generativa (que explica) fue
   la decisión de diseño correcta: Claude nunca inventa un número, y el
   sistema sigue funcionando por completo si la API de Claude no está
   disponible.
5. Un mecanismo de reentrenamiento simple, con una regla de promoción
   explícita, es suficiente para un sistema de este tamaño — y **una prueba
   real** demostró que protege el sistema en vez de degradarlo con datos
   todavía escasos.

## 16. Limitaciones

- **ERA5 es reanálisis, no observación directa** — para la temperatura la
  fidelidad es muy alta (r = 0,985 frente a estación real), pero para el
  viento es notablemente menor (r = 0,72); cualquier conclusión sobre viento
  debe leerse con ese matiz.
- **Una sola ubicación.** Los regímenes, el modelo y sus hiperparámetros no
  generalizan directamente a climas distintos; el diseño de base de datos y
  código sí soporta múltiples ubicaciones (trabajo futuro).
- **Volumen de datos en tiempo real todavía bajo** (semanas, no meses desde
  el despliegue) — el reentrenamiento se ha demostrado funcionalmente, no
  con una mejora numérica grande, tal como se anticipó en el diseño.
- **El reloj del entorno de desarrollo** puede desincronizarse ligeramente
  del real; se mitigó acotando la antigüedad de la observación a valores no
  negativos, pero es una fragilidad a vigilar en despliegue real.
- **Horizonte de predicción limitado a 24 h** — más allá, la temperatura
  deja de ser razonablemente predecible con las variables usadas aquí sin
  un modelo físico de circulación atmosférica.
- **La interpretación de Claude no se ha podido probar con la API real** en
  este entorno de desarrollo (sin clave configurada) — el *fallback* se
  verificó exhaustivamente, pero la calidad del texto generado en producción
  queda pendiente de una primera ejecución con clave activa.

## 17. Trabajo futuro

1. **Multi-ubicación:** extender el sistema (el esquema ya lo soporta) a
   varias ciudades para comparar regímenes y modelos entre climas distintos.
2. **Ventana de entrenamiento reciente vs. completa:** comparar el modelo
   entrenado con los 26 años completos frente a uno entrenado solo con la
   última década, dado el calentamiento observado (~+1 °C 2000→2025,
   Fase 2) — podría mejorar la predicción a costa de perder patrones raros.
3. **Activar la interpretación con Claude en producción** con la clave real
   y evaluar la calidad y utilidad del texto generado con usuarios reales.
4. **Ampliar las variables objetivo:** predecir precipitación o viento, no
   solo temperatura, con la misma disciplina de validación.
5. **Panel de reentrenamiento en el dashboard**, mostrando el historial de
   `retraining_runs` visualmente (el *endpoint* ya existe:
   `GET /api/model/retraining-history`).
6. **Alertas proactivas** cuando se detecta una anomalía relevante, en vez
   de solo mostrarla en el dashboard.

---

## 18. Referencias y fuentes de datos

- Open-Meteo — [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5, CC BY 4.0) y [Forecast API](https://open-meteo.com/)
- Meteostat — [dev.meteostat.net](https://dev.meteostat.net/) (CC BY 4.0)
- Anthropic — [Claude API](https://docs.anthropic.com/) (`claude-haiku-4-5`)
- ECMWF/Copernicus — reanálisis ERA5 (fuente subyacente de Open-Meteo Historical)

## Anexo — Documentación técnica por fase

| Fase | Documento |
|---|---|
| 1 | [`fase-1-analisis-y-arquitectura.md`](fase-1-analisis-y-arquitectura.md) |
| 2 | [`fase-2-data-understanding.md`](fase-2-data-understanding.md) · [`diccionario-de-datos.md`](diccionario-de-datos.md) |
| 3 | [`fase-3-data-preparation.md`](fase-3-data-preparation.md) · [`fase-3-eda.md`](fase-3-eda.md) |
| 4 | [`fase-4-clustering.md`](fase-4-clustering.md) · [`fase-4-anomalias.md`](fase-4-anomalias.md) |
| 5 | [`fase-5-modelado.md`](fase-5-modelado.md) |
| 6 | [`fase-6-backend-api.md`](fase-6-backend-api.md) |
| 7 | [`fase-7-integracion-claude.md`](fase-7-integracion-claude.md) |
| 8 | [`fase-8-frontend.md`](fase-8-frontend.md) |
| 9 | [`fase-9-retraining.md`](fase-9-retraining.md) |

Reproducibilidad, estructura del repositorio y comandos de arranque:
[`README.md`](../README.md) del proyecto.
