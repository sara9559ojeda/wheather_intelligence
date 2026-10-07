# FASE 9 — Retraining + robustez

**Ubicación:** Madrid-Barajas · **Fecha:** 2026-09-14
**Reproducible:** `python -m backend.ml.modeling.run_retraining [--force|--check-only]`

---

## 1. El problema: un modelo entrenado una vez se queda anticuado

Los modelos de la Fase 5 se entrenaron con datos hasta el 5 de septiembre de
2026. El sistema sigue recibiendo observaciones nuevas cada 30 minutos (Fase
6) — pero **nada las usa para mejorar el modelo** hasta ahora. Con el tiempo,
el mundo cambia (estacionalidad distinta, instrumentación, clima) y el modelo
puede quedarse desactualizado (*model drift*).

**Por qué no basta con "reentrenar cada noche y ya está":** un reentrenamiento
automático sin control puede **empeorar** el sistema — menos datos de una
racha rara, un `random_state` con mala suerte, un cambio de distribución. La
Fase 1 (§17) pidió explícitamente evitar MLOps pesado (nada de Airflow/MLflow)
pero exigió una **regla de promoción explícita**: el modelo nuevo sustituye al
viejo **solo si de verdad es mejor**, medido de forma justa.

**Intuición:** es como un aspirante a un puesto de trabajo: no lo contratas
solo por presentarse. Le haces la **misma prueba** que le hiciste al que ya
está en el puesto, y solo lo contratas si la supera con margen.

---

## 2. Disparador — ¿cuándo tocaría reentrenar?

Dos condiciones (fase-1 §17), cualquiera dispara:

| Disparador | Umbral | Config |
|---|---|---|
| **Volumen** | ≥ `RETRAIN_VOLUME_TRIGGER` observaciones `realtime` nuevas desde el último reentrenamiento | 1000 (por defecto) |
| **Calendario** | ≥ `RETRAIN_CALENDAR_DAYS` días desde la última comprobación (o nunca se ha reentrenado) | 30 días |

El *cutoff* de "última vez" se guarda en la propia tabla `retraining_runs`
(`data_cutoff`); si nunca se ha reentrenado, se usa el final del histórico
original. Comprobarlo es una consulta barata — el scheduler lo hace cada
`RETRAIN_CHECK_INTERVAL_HOURS` (24 h por defecto); el reentrenamiento en sí
(varios minutos) solo ocurre cuando de verdad toca.

---

## 3. Procedimiento

```
weather_observations (histórico + tiempo real acumulado)
   -> limpieza (backend.ml.preparation.clean)         -> aborta si la calidad falla
   -> features + targets (mismo código que la fase 3/5)
   -> partición cronológica 70/15/15 (mismas proporciones, fase 3)
   -> por cada horizonte (1,3,6,12,24 h):
        entrenar un RETADOR (baselines + Ridge/RF/HistGB, igual que fase 5)
        evaluar al CAMPEÓN ACTIVO sobre el MISMO test nuevo (comparación justa)
        decide_promotion(...)  -> promocionar o conservar
   -> registrar el intento completo en retraining_runs (se promocione o no)
```

### 3.1 Puerta de calidad (aborta antes de tocar nada)

| Chequeo | Umbral de aborto |
|---|---|
| % de valores recortados a límite físico (`clean_observations`) | > 5 % |
| % de timestamps duplicados | > 1 % |

Si se supera, el intento queda registrado con `status = "aborted_validation"`
y **no se toca ningún modelo ni artefacto**.

### 3.2 Regla de promoción (escrita, determinista — `decide_promotion`)

```
promover el retador SI
    RMSE_retador <= RMSE_campeón_en_test_nuevo × (1 − epsilon)      [mejora clara]
 O  RMSE_campeón_en_test_nuevo > RMSE_campeón_original × degradación
    Y RMSE_retador <= RMSE_campeón_en_test_nuevo                     [campeón ha
                                                                        degradado y
                                                                        el retador no
                                                                        es peor]
```

- `epsilon = 0.02` (2 % de mejora exigida) — evita promocionar por ruido de
  entrenamiento.
- `degradación = 1.20` (el campeón ha empeorado un 20 % desde que se entrenó)
  — cubre el caso de *drift* real aunque el retador no sea brillante, siempre
  que no sea peor que el campeón ya deteriorado.
- Si no hay campeón activo evaluable (primera vez, o artefacto perdido) se
  promociona directamente.

**El campeón se reevalúa sobre el test NUEVO**, no se reutiliza su métrica
antigua — comparar RMSE de dos test distintos no sería una comparación justa.

### 3.3 Si se promociona

1. El artefacto anterior se **archiva** (`backend/ml/artifacts/archive/model_temp_h{H}_<timestamp>.joblib`), nunca se borra.
2. Se guarda el nuevo artefacto en su sitio.
3. Se desactiva la fila `model_runs` del campeón anterior y se inserta una
   nueva con `is_active=true`, métricas de validación/test/CV/importancia.
4. Se recalculan las `predictions` del nuevo campeón sobre el test nuevo.
5. Si el backend está corriendo, se emite un evento SSE (`model_updated`).

### 3.4 Si NO se promociona

El intento del retador **también se registra** (fila `model_runs` con
`is_active=false`) — trazabilidad completa de qué se probó y por qué se
descartó, sin tocar el modelo en producción.

---

## 4. Ejecución real (verificación, no una simulación)

```bash
python -m backend.ml.modeling.run_retraining --check-only
# Disparador: should_run=True reason=calendar n_new=210 desde=2026-09-05 23:00:00+00:00

python -m backend.ml.modeling.run_retraining
```

**Resultado real** (id=1, `trigger_reason=calendar`, 234.090 filas consolidadas
— 233.880 históricas + 210 nuevas de tiempo real —, calidad perfecta: 0
duplicados, 0 recortes; duración **283,7 s**):

| Horizonte | Retador (RMSE test) | Campeón en el test nuevo | Campeón original | Decisión |
|---|---:|---:|---:|---|
| 1 h | 0,724 | 0,722 | 0,725 | conservado |
| 3 h | 1,215 | 1,214 | 1,216 | conservado |
| 6 h | 1,558 | 1,556 | 1,557 | conservado |
| 12 h | 1,876 | 1,875 | 1,873 | conservado |
| 24 h | 2,168 | 2,167 | 2,164 | conservado |

`status = not_promoted`. En los 5 horizontes el retador queda **a centésimas**
del campeón (diferencias de 0,001–0,003 °C de RMSE, ruido de entrenamiento) —
muy por debajo del 2 % de mejora exigido, y el campeón no ha degradado
(compárese "campeón en el test nuevo" con "campeón original": prácticamente
idénticos). El sistema **conserva correctamente** los 5 campeones de la fase 5.

**Lectura (la que predijo la fase 1, C6):** con solo ~210 observaciones nuevas
sobre 234.000 (~0,1 % del dataset), era esperable que el retador no
desplazara a un modelo entrenado con 18 años de historia. El valor de esta
fase no está en el número — está en que el mecanismo **compara de forma
justa, decide con una regla escrita y deja rastro completo**, exactamente lo
que hará dentro de unos meses cuando la ingesta en tiempo real acumule
volumen real, sin cambiar una línea de código. Las 5 filas nuevas de
`model_runs` (retadores descartados) quedan igualmente guardadas para
trazabilidad — ver `results` en `GET /api/model/retraining-history`.

---

## 5. Persistencia — tabla `retraining_runs`

| Columna | Contenido |
|---|---|
| `trigger_reason` | `volume` \| `calendar` \| `manual` |
| `n_new_observations` | observaciones nuevas que motivaron el intento |
| `data_cutoff` | hasta qué instante se consolidaron datos en este intento |
| `status` | `promoted` \| `not_promoted` \| `aborted_validation` \| `aborted_error` |
| `validation_report` (JSONB) | filas, duplicados, valores recortados |
| `results` (JSONB) | por horizonte: tipo de retador, RMSE de ambos, decisión y motivo |
| `duration_seconds` | cuánto tardó |

Endpoint: `GET /api/model/retraining-history`.

---

## 6. Robustez (más allá del reentrenamiento)

- **El job nunca tumba el scheduler**: `retrain_check_job` captura cualquier
  excepción, hace `rollback()` y sigue — igual que `ingest_job` (fase 6).
- **Nada se borra**: artefactos archivados, filas de `model_runs` de retadores
  descartados conservadas.
- **Reproducible**: mismo `random_state=42` en todos los modelos; la
  partición cronológica usa las mismas proporciones y el mismo hueco anti-fuga
  que la fase 3.
- **Coste acotado**: comprobar el disparador es una consulta; reentrenar de
  verdad son minutos, como mucho una vez al mes en la práctica.
- **Tests**: `backend/tests/test_retrain.py` — 8 pruebas puras de
  `decide_promotion` (mejora clara, por debajo de epsilon, frontera exacta,
  sin campeón previo, degradación con y sin retador competente, degradación
  leve que no dispara, valores por defecto de `settings`) + 3 con base de
  datos real (disparador, conteo de observaciones nuevas, y que la puntuación
  del campeón activo sobre su propio test coincide en orden de magnitud con
  la métrica ya guardada). La ejecución completa (varios minutos) se verifica
  a mano, no en la batería automática — igual que el resto de *pipelines* de
  entrenamiento del proyecto.

---

## Q-gate fase 9 → fase 10

✅ disparador por volumen o calendario · puerta de calidad antes de reentrenar
· partición cronológica reproducida con las mismas proporciones · comparación
campeón-retador sobre el mismo test · regla de promoción escrita y
determinista · artefactos archivados, nunca borrados · trazabilidad completa
en `retraining_runs` (se promocione o no) · scheduler robusto ante errores ·
11 pruebas nuevas en verde.

**Siguiente (fase 10):** documentación académica final — planteamiento,
justificación, objetivos, metodología CRISP-ML(Q) completa, resultados,
conclusiones, limitaciones y trabajo futuro, lista para la defensa.
