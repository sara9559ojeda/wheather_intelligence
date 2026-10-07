# FASE 3 — Data Preparation

**Ubicación:** Madrid-Barajas · **Fecha:** 2026-09-08
**Pipeline reproducible:** `python -m backend.ml.preparation.build_dataset`
**Salida:** `data/processed/{train,valid,test}.parquet` + `dataset_manifest.json`

El EDA está en un documento aparte: [`fase-3-eda.md`](fase-3-eda.md).

---

## 1. Flujo

```
weather_observations (PostgreSQL, source='historical')
   → limpieza            (backend/ml/preparation/clean.py)
   → features derivadas  (backend/app/services/features/builder.py)
   → targets (t + H)     (add_targets)
   → partición cronológica 70/15/15 con hueco de 24 h  (backend/ml/preparation/split.py)
   → data/processed/
```

**Por qué el constructor de features vive en `backend/app/services/features/` y no en `backend/ml/`:**
las features de *entrenamiento* y las de *inferencia en tiempo real* (fase 6) deben
construirse con **exactamente el mismo código**. Si se duplicara la lógica, cualquier
diferencia mínima (una unidad, un desfase de una hora) provocaría *train-serving skew*:
el modelo vería en producción variables distintas a las que aprendió.

## 2. Limpieza

`clean_observations()` corrige **solo errores inequívocos** (de unidad, signo o
sensor) recortando a límites físicos duros: humedad y nubosidad a `[0, 100]`,
precipitación / radiación / viento a `≥ 0`, dirección del viento a `[0, 360]`.
**Nunca** recorta extremos meteorológicos reales.

| Métrica | Valor |
|---|---|
| Filas de entrada | 233 880 |
| Duplicados de timestamp eliminados | **0** |
| Valores recortados a límite físico | **0** |
| Filas de salida | 233 880 |

Resultado esperado: ERA5 es un reanálisis físicamente consistente (ver
[`fase-2-data-understanding.md`](fase-2-data-understanding.md)), así que no hay
nada que corregir. El módulo existe para los datos en tiempo real y para el
contraste de estación, que sí traen ruido (p. ej. humedad > 100 % en Meteostat).

## 3. Feature engineering — 38 features, con justificación

### 3.1 Calendario y codificación cíclica
`year`, `month`, `day`, `hour`, `dayofweek`, `dayofyear`, `is_weekend`.

Además **codificación cíclica** de la hora y del día del año:
`hour_sin = sin(2π·hora/24)`, `hour_cos = cos(2π·hora/24)`, e igual para `dayofyear/365.25`.

> **Justificación:** para un modelo, "hora = 23" y "hora = 0" están muy lejos
> numéricamente, cuando en realidad son consecutivas. Al proyectarlas sobre un
> círculo (sin, cos), la medianoche queda pegada a la 1:00. El EDA muestra que la
> temperatura está dominada por el ciclo diario y el anual → estas 4 variables
> deberían ser de las más informativas.

### 3.2 Dirección del viento → `wind_dir_sin`, `wind_dir_cos`
Misma idea: `wind_direction_10m` es una variable **circular** (0° = 360° = Norte).
Usarla en grados haría creer al modelo que un viento del Norte (0°) y otro casi
del Norte (359°) son opuestos.

### 3.3 Retardos (*lags*) — solo pasado
| Variable | Retardos (horas) | Por qué |
|---|---|---|
| `temperature_2m` | 1, 2, 3, 6, 24 | Inercia térmica + ciclo diario (lag 24 h ≈ "misma hora ayer") |
| `relative_humidity_2m` | 1, 3, 24 | Evolución de la humedad |
| `surface_pressure` | 1, 3 | Estado del sistema sinóptico |
| `wind_speed_10m` | 1, 3 | Persistencia del viento |
| `shortwave_radiation` | 1, 24 | Radiación reciente y de la víspera |
| `cloud_cover` | 1 | Nubosidad reciente |

Cada `lag_Nh` en el instante `t` es el valor en `t − N`. **Ningún lag usa el
futuro** (test en `backend/tests/test_features.py`).

### 3.4 Medias / desviaciones móviles (ventana **hacia atrás**, sin el instante actual)
`temperature_2m_roll_{mean,std}_{3h,24h}`, `wind_speed_10m_roll_mean_3h`,
`precipitation_roll_sum_{3h,24h}`.

> **Detalle anti-fuga:** la ventana se calcula sobre `serie.shift(1).rolling(N)`,
> de modo que en `t` la media cubre `{t−N, …, t−1}` y **nunca incluye `t`**.
> La `std` móvil captura cuán "turbulentas" han sido las últimas horas; los
> acumulados de precipitación resumen una variable que es 0 el 95 % del tiempo.

### 3.5 Deltas (tendencias)
`temperature_2m_delta_1h` (= T_t − T_{t−1}), `relative_humidity_2m_delta_1h`,
`surface_pressure_delta_3h`.

> **Justificación:** la **tendencia barométrica de 3 h** es un predictor clásico
> de cambio de tiempo (presión que cae rápido → frente en camino).

### 3.6 Binarias
`is_raining` (= `precipitation > 0`), `is_day` (0.0/1.0).

> **Justificación:** la precipitación está tan concentrada en 0 que una señal
> binaria "llueve / no llueve" es más estable que el valor crudo.

### 3.7 Filas descartadas
Las primeras **24 filas** (2000-01-01 00:00 → 23:00) no tienen historia
suficiente para los lags/rollings de 24 h → se eliminan.
`233 880 → 233 856` filas.

## 4. Targets

Para cada horizonte `H ∈ {1, 3, 6, 12, 24}` se crea
`target_temp_h{H} = temperature_2m.shift(-H)` (la temperatura **H horas en el
futuro**). Las últimas `H` filas de la serie no tienen target para ese horizonte
(quedan NaN) y se descartan **por horizonte** en el entrenamiento (fase 5), no aquí.

## 5. Partición cronológica (anti *data leakage*)

**Nada de particiones aleatorias.** Los datos son una serie temporal: barajar
metería horas futuras en el conjunto de entrenamiento y las métricas saldrían
optimistas y falsas (ver [`fase-1`](fase-1-analisis-y-arquitectura.md) §15).

Se corta **por orden de tiempo** en 70 % / 15 % / 15 % y se deja un **hueco de
24 h** (= horizonte máximo) entre tramos, para que los targets del final de un
tramo (temperatura hasta 24 h después) no caigan dentro del tramo siguiente.

| Tramo | Filas | Desde | Hasta |
|---|---:|---|---|
| **train** | 163 699 | 2000-01-02 00:00 UTC | 2018-09-04 18:00 UTC |
| *(hueco 24 h)* | | | |
| **valid** | 35 078 | 2018-09-05 19:00 UTC | 2022-09-06 08:00 UTC |
| *(hueco 24 h)* | | | |
| **test** | 35 031 | 2022-09-07 09:00 UTC | 2026-09-05 23:00 UTC |

- **test = los ~4 años más recientes** → mide el modelo en condiciones "de hoy".
- El EDA (fase 3) se hizo sobre **train + valid**; **test no se ha mirado** y no
  se tocará hasta la evaluación final (fase 5).

## 6. Lo que **no** se hace en esta fase (y por qué)

| Paso | Cuándo | Por qué no ahora |
|---|---|---|
| **Escalado / normalización** | Fase 5, dentro de un `Pipeline` de sklearn | El `scaler` debe ajustarse **solo con train**. Hacerlo aquí, sobre todo el dataset, filtraría la media/σ del futuro. |
| **Imputación de faltantes** | No necesaria | ERA5 no tiene nulos. |
| **Selección de variables** | Fase 5 | Se decide con la importancia del Random Forest y la colinealidad (p. ej. `wind_speed` ↔ `wind_gusts`, r = 0.90). |

## 7. Comprobaciones automáticas (tests)

`backend/tests/test_features.py` y `test_split.py`:

- `test_lag_uses_only_the_past` — `lag_1h[t] == valor[t−1]`, `lag_24h[t] == valor[t−24]`.
- `test_rolling_window_excludes_current_instant` — la media móvil en `t` no incluye `t`.
- `test_cyclical_encoding_is_on_unit_circle` — `sin² + cos² = 1`.
- `test_targets_point_to_the_future` — `target_temp_h{H}[t] == temperature_2m[t+H]`.
- **`test_future_values_do_not_leak_into_present_features`** — corromper observaciones
  futuras **no** altera las features de filas anteriores.
- `test_gap_in_series_produces_nan_lags_not_wrong_values` — un hueco en la serie
  genera lags NaN, no valores de la hora equivocada.
- `test_splits_are_ordered_in_time_with_a_gap`, `test_no_timestamp_appears_in_two_splits`,
  `test_train_is_the_oldest_data`.

---

## Q-gate fase 3 → fase 4

✅ Pipeline reproducible de un comando · 38 features justificadas · partición
cronológica congelada en `split_meta.json` · tests anti-*leakage* en verde · EDA
completo con interpretación escrita.

**Siguiente (fase 4):** minería de datos — *clustering* K-Means (Elbow +
Silhouette) para descubrir regímenes meteorológicos, e Isolation Forest sobre
residuales des-estacionalizados para detección de anomalías.
