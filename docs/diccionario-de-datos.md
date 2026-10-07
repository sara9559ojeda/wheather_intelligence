# Diccionario de datos — Weather Intelligence

**Ubicación:** Madrid-Barajas (aeropuerto Adolfo Suárez, LEMD) — lat 40.4936, lon −3.5668
**Zona horaria de almacenamiento:** UTC (la conversión a hora local se hace solo en el frontend)
**Última actualización:** 2026-09-08 (fase 2)

Este documento describe **las dos fuentes de datos crudas** tal y como se
descargan en `data/raw/`. Las variables *derivadas* (lags, medias móviles,
codificación cíclica, etc.) se documentan en la fase 3 (Data Preparation).

Las cifras (rangos observados, % de nulos) provienen del informe reproducible
[`fase-2-data-understanding.md`](fase-2-data-understanding.md), generado por
`python -m backend.ml.data_understanding`.

---

## 1. Dataset histórico principal — `open_meteo_madrid_barajas_hourly.parquet`

| Atributo | Valor |
|---|---|
| Fuente | Open-Meteo Historical Weather API — reanálisis **ERA5** (ECMWF / Copernicus) |
| URL | `https://archive-api.open-meteo.com/v1/archive` |
| Licencia | **CC BY 4.0** (atribución obligatoria) |
| Método | 1 petición por año natural, formato CSV, `timezone=UTC` |
| Registros | **233 880** filas horarias |
| Periodo | 2000-01-01 00:00 → 2026-09-05 23:00 UTC |
| Frecuencia | Horaria, **sin huecos** (cobertura 100.0 %), **sin duplicados**, **sin nulos** |
| Celda de rejilla real | lat 40.527, lon −3.559, **elevación 618 m** (ERA5 ajusta a su rejilla ~25 km) |
| Granularidad espacial | 1 punto (la rejilla ERA5 promedia el terreno en ~25 km) |
| Naturaleza | **Reanálisis** (modelo físico que asimila observaciones), *no* medición directa |

### 1.1 Variables

Nombre = nombre canónico del proyecto (el sufijo con unidades del CSV de
Open-Meteo se elimina al descargar). Rango observado = [mín, máx] en los datos.

| Variable | Unidad | Tipo | Rol | Descripción | Rango plausible | Rango observado | Notas |
|---|---|---|---|---|---|---|---|
| `timestamp` | ISO-8601 UTC | datetime (tz-aware) | índice | Instante de la observación, inicio de la hora | — | 2000→2026 | Clave temporal; única |
| `temperature_2m` | °C | float | **objetivo** + predictora (lags) | Temperatura del aire a 2 m | [−25, 48] | **[−10.0, 40.7]** | Media 14.7; std 9.0. Variable central del proyecto |
| `relative_humidity_2m` | % | int | predictora / contexto | Humedad relativa a 2 m | [0, 100] | [6, 100] | Media 59; correlación esperada negativa con temperatura |
| `dew_point_2m` | °C | float | predictora | Punto de rocío a 2 m | [−35, 30] | [−19.8, 23.4] | Se deriva de T+HR → **no usar para predecir T del mismo instante** (fuga) |
| `apparent_temperature` | °C | float | contexto (dashboard) | Sensación térmica (viento + humedad + radiación) | [−35, 55] | [−14.6, 41.6] | Se deriva de T → **solo contexto**, nunca predictora de T instantánea |
| `surface_pressure` | hPa | float | predictora | Presión atmosférica a la altura de la estación | [905, 972] | [910.7, 969.7] | Media 946 (coherente con 618 m). 4 valores >968 = anticiclón real |
| `pressure_msl` | hPa | float | predictora / contraste | Presión reducida a nivel del mar | [975, 1050] | [981.9, 1044.9] | Comparable con la presión de la estación Meteostat |
| `precipitation` | mm | float | predictora / contexto | Precipitación total de la hora (lluvia + nieve fundida) | [0, 60] | [0, 26.7] | **Muy sesgada a 0** (p75 = 0; p99 = 1.3). Considerar variable binaria "llueve" |
| `rain` | mm | float | contexto | Precipitación líquida de la hora | [0, 60] | [0, 26.7] | Desglose de `precipitation` |
| `snowfall` | cm | float | contexto | Nieve de la hora | [0, 30] | [0, 1.96] | Casi siempre 0 en Madrid |
| `cloud_cover` | % | int | predictora | Nubosidad total | [0, 100] | [0, 100] | Media 44; muy bimodal (despejado / cubierto) |
| `cloud_cover_low` | % | int | predictora (opc.) | Nubosidad baja | [0, 100] | [0, 100] | — |
| `cloud_cover_mid` | % | int | predictora (opc.) | Nubosidad media | [0, 100] | [0, 100] | — |
| `cloud_cover_high` | % | int | predictora (opc.) | Nubosidad alta | [0, 100] | [0, 100] | — |
| `wind_speed_10m` | km/h | float | predictora | Velocidad del viento a 10 m | [0, 130] | [0, 40.9] | Media 8.6 |
| `wind_direction_10m` | ° | int | predictora | Dirección del viento a 10 m (0 = N, 90 = E) | [0, 360] | [0, 360] | **Variable circular** → codificar como `sin`/`cos` en fase 3 |
| `wind_gusts_10m` | km/h | float | contexto | Racha máxima a 10 m | [0, 180] | [0.7, 101.2] | — |
| `wind_speed_100m` | km/h | float | predictora (opc.) | Velocidad del viento a 100 m | [0, 160] | [0, 65.7] | Útil como proxy de estabilidad atmosférica |
| `shortwave_radiation` | W/m² | float | **predictora fuerte** | Irradiancia solar global horizontal | [0, 1100] | [0, 1031] | 0 de noche; muy ligada a hora solar y nubosidad |
| `direct_radiation` | W/m² | float | predictora (opc.) | Componente directa | [0, 1000] | [0, 906] | — |
| `diffuse_radiation` | W/m² | float | predictora (opc.) | Componente difusa | [0, 750] | [0, 469] | — |
| `et0_fao_evapotranspiration` | mm | float | predictora (opc.) | Evapotranspiración de referencia FAO-56 | [−0.2, 2.5] | [0, 0.97] | Índice compuesto; redundante con radiación |
| `weather_code` | código WMO | int | contexto (iconos) | Condición meteorológica (WW) | conjunto WMO | 13 valores: 0,1,2,3,51,53,55,61,63,65,71,73,75 | Mapear a etiquetas; *one-hot* si entra al modelo |
| `is_day` | 0/1 | int | predictora | 1 = hay luz solar en ese instante | {0, 1} | {0, 1} | Señal día/noche barata y potente |

> **Variable objetivo del modelo supervisado:** `temperature_2m` en `t + H`,
> con `H ∈ {1, 3, 6, 12, 24}` horas (fase 5).

---

## 2. Dataset de contraste — `meteostat_08221_hourly.parquet`

| Atributo | Valor |
|---|---|
| Fuente | **Meteostat**, estación **08221** "Madrid / Barajas" |
| Identificadores | WMO 08221 · ICAO **LEMD** · IATA MAD · GHCN SPE00120278 |
| Ubicación estación | lat 40.45, lon −3.55, **elevación 609 m** |
| Origen de los datos | Informes METAR/SYNOP agregados (NOAA, AEMET, DWD…) |
| Licencia | **CC BY 4.0** |
| Registros | **228 281** filas horarias |
| Periodo | 2000-01-01 → 2026-03-29 UTC |
| Cobertura | **99.24 %** (1 758 horas ausentes) — datos reales, con huecos |
| Uso en el proyecto | **Contraste** de fidelidad de ERA5. No entra al modelo |

### 2.1 Variables (nombres originales de Meteostat, sin renombrar en el crudo)

| Columna | Nombre descriptivo | Unidad | % nulos | Rango observado | Notas |
|---|---|---|---|---|---|
| `timestamp` | Instante (UTC) | ISO-8601 | 0 | 2000→2026 | — |
| `temp` | Temperatura del aire | °C | 0.12 % | [−12.7, 42.4] | Comparable con `temperature_2m` de ERA5 |
| `dwpt` | Punto de rocío | °C | 0.16 % | [−22, 24] | — |
| `rhum` | Humedad relativa | % | 0.16 % | [5, **152**] | ⚠️ **31 valores > 100 %** (error de la fuente) → recortar en fase 3 |
| `prcp` | Precipitación horaria | mm | **85.6 %** | [0, 13.9] | Muy incompleta; no fiable para análisis serio |
| `snow` | Espesor de nieve | mm | **98.9 %** | [0, 100] | Prácticamente vacía |
| `wdir` | Dirección del viento | ° | 7.1 % | [0, 360] | Circular |
| `wspd` | Velocidad del viento | km/h | 0.25 % | [0, 87.1] | Comparable con `wind_speed_10m` de ERA5 |
| `wpgt` | Racha máxima | km/h | **90.8 %** | [0, 64.8] | Muy incompleta |
| `pres` | Presión a nivel del mar | hPa | 15.8 % | [980.5, 1047.4] | Comparable con `pressure_msl` de ERA5 |
| `tsun` | Minutos de sol / hora | min | **100 %** | — | **Columna vacía → descartar** |
| `coco` | Código de condición Meteostat | código | 69.4 % | [0, 26] | Distinto del código WMO de Open-Meteo |

### 2.2 Contraste ERA5 ↔ estación (horas coincidentes, 2000–2026)

| Variable | n horas | Sesgo (ERA5 − est.) | MAE | RMSE | Correlación Pearson |
|---|---:|---:|---:|---:|---:|
| Temperatura (°C) | 228 015 | −0.55 | **1.31** | 1.68 | **0.985** |
| Punto de rocío (°C) | 227 921 | −0.12 | 1.34 | 1.82 | 0.934 |
| Humedad relativa (%) | 227 921 | +1.08 | 6.19 | 8.44 | 0.941 |
| Velocidad viento (km/h) | 227 706 | −1.75 | 4.26 | 5.70 | **0.721** |
| Presión nivel del mar (hPa) | 192 232 | +0.31 | 1.05 | 1.36 | 0.986 |

**Lectura:**
- ERA5 representa **muy bien** la temperatura y la presión de esta ubicación
  (correlación ≈ 0.99; MAE de temperatura ≈ 1.3 °C, del orden de la resolución
  del propio dato).
- El **viento** es la variable con peor acuerdo (correlación 0.72; ERA5 tiende a
  subestimar rachas locales) — coincide con lo previsto en la fase 1 (§7,
  limitaciones de ERA5). Se tendrá en cuenta al interpretar los modelos y las
  anomalías que dependan del viento.
- La precipitación no se contrasta: la columna `prcp` de la estación está vacía
  en un 86 % de las horas.

---

## 3. Problemas de calidad conocidos (entrada a la fase 3)

| # | Dataset | Problema | Acción prevista en Data Preparation |
|---|---|---|---|
| Q1 | ERA5 | 4 valores de `surface_pressure` ligeramente por encima de 968 hPa | Verificar que son anticiclones reales (probable) y ampliar el rango, no recortar |
| Q2 | Estación | 31 valores de `rhum` > 100 % (máx 152) | Recortar a 100 (`clip`) y registrar cuántos |
| Q3 | Estación | `tsun` 100 % nula; `snow`, `wpgt`, `prcp` > 85 % nulas | Descartar `tsun`; no usar las demás para análisis, solo mención |
| Q4 | Estación | 1 758 horas ausentes (0.76 %) | Documentar; el dataset de estación no se imputa (solo se usa para contraste) |
| Q5 | ERA5 | `precipitation` fuertemente sesgada a cero | Crear variable binaria `is_raining` + acumulados móviles; no normalizar ingenuamente |
| Q6 | ERA5 | `wind_direction_10m` es circular (0° = 360°) | Codificar `wind_dir_sin`, `wind_dir_cos` |
| Q7 | Ambos | Tendencia de calentamiento ~+1 °C entre 2000 y 2025 (ver figura anual) | Entrenar el modelo también con ventana reciente (2015→) y comparar (fase 1, C4) |

---

## 4. Trazabilidad

Cada archivo de `data/raw/` tiene un `*.meta.json` versionado en git con:
`source`, `license`, `downloaded_at_utc`, `date_range_utc`, `n_rows`, `columns`,
`missing_values_per_column`, `duplicate_timestamps` y **`sha256`** del archivo.

SHA-256 del dataset principal (descarga del 2026-09-08):
`54473c67b3c41870a23a902efd37cec57cb1c90f7c48eb6cfa6ffacfec30f593`
