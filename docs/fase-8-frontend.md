# FASE 8 — Frontend (React + Tailwind + Recharts)

**Ubicación:** Madrid-Barajas · **Fecha:** 2026-09-08
**Stack:** Vite 6 · React 18 · TypeScript · Tailwind CSS 4 · Recharts · TanStack Query
**Arranque:** `npm --prefix frontend run dev` → http://localhost:5173

---

## 1. Principios de diseño

Dashboard de datos, estética técnica y contenida (referencia: interfaces de
producto de infraestructura de datos). Un solo color de acento + semánticos
(ok / warn / danger). Tema oscuro. Jerarquía por **tipografía y espacio**, no por
color. *Design tokens* en `src/index.css` (`@theme`), sin valores mágicos
repartidos por los componentes.

**Jerarquía del dashboard** (no "20 tarjetas iguales"):

```
1. Condiciones actuales  +  Predicción           ← lo más importante, arriba
2. Comparación histórica  +  Regímenes (clusters)
3. Evolución temporal (serie larga)
4. Temp vs humedad  +  Matriz de correlación
5. Anomalías  +  Predicho vs real
6. Rendimiento del modelo
7. Interpretación con IA (Claude)
```

## 2. Las 8 visualizaciones (fase 1, §22)

| # | Visualización | Componente | Fuente |
|---|---|---|---|
| 1 | Temperatura vs tiempo | `HistoryCharts` (toggle) | `/weather/history` |
| 2 | Humedad vs tiempo | `HistoryCharts` (toggle) | `/weather/history` |
| 3 | Temperatura vs humedad | `ScatterTempHumidity` (+ r de Pearson) | `/weather/history` |
| 4 | Matriz de correlación | `CorrelationHeatmap` | `/stats/correlations` |
| 5 | Clusters / regímenes | `ClustersPanel` (barras + centroides) | `/dashboard` |
| 6 | Anomalías | `AnomaliesPanel` (dispersión de score + tabla) | `/anomalies` |
| 7 | Predicho vs real | `PredictedVsActual` (selector de horizonte) | `/predictions/history` |
| 8 | Histórico vs actual | `HistoricalComparison` (barras de percentil) | `/dashboard` |

Cada gráfico incluye ejes mínimos, tooltip, unidades consistentes y su matiz
("correlación no implica causalidad", "una anomalía no implica fenómeno extremo",
"predicción con incertidumbre").

## 3. Actualización en vivo (SSE)

`src/hooks/useSSE.ts` abre `EventSource("/api/stream")`. Al recibir un evento
`update` (observación nueva en el backend) **invalida** las queries de TanStack
Query afectadas → *refetch* automático. El indicador del cabecera muestra el
estado de la conexión (en vivo / conectando / sin conexión) y cuándo fue la
última actualización.

> Patrón "SSE notifica, REST trae los datos": el evento no lleva payload útil,
> solo dispara el *refetch*. `refetchInterval` de 5 min actúa como red de
> seguridad si el SSE cae.

## 4. Estados

- **Carga**: *skeletons* (no pantallas en blanco).
- **Error**: mensaje + botón "Reintentar".
- **Vacío**: p. ej. AI Insights sin clave → explica qué falta y que el resto
  funciona igual; predicción sin datos recientes → explica por qué.

## 5. Accesibilidad

HTML semántico (`<header>`, `<main>`, `<section>`, `<footer>`, `<table>`),
jerarquía de encabezados, `:focus-visible` visible, `prefers-reduced-motion`
respetado, contraste alto, ninguna información transmitida solo por color
(los badges llevan texto, las barras de percentil llevan el número).

## 6. Arquitectura

```
frontend/src/
  main.tsx              QueryClientProvider
  App.tsx               layout + composición de las filas
  index.css             design tokens (@theme) + reset
  api/
    types.ts            tipos que reflejan backend/app/schemas
    client.ts           fetch tipado + ApiError
    hooks.ts            un hook useQuery por endpoint
  hooks/useSSE.ts       EventSource -> invalidación de queries
  lib/format.ts         unidades, fechas, números (locale es-ES)
  components/
    ui.tsx              Card, Skeleton, ErrorState, EmptyState, Metric, Badge, tokens de gráfico
    CurrentConditions / PredictionPanel / HistoricalComparison / ClustersPanel
    HistoryCharts / ScatterTempHumidity / CorrelationHeatmap
    AnomaliesPanel / PredictedVsActual / ModelPerformance / AiInsights
```

Lógica de negocio fuera de los componentes: **el frontend solo pinta**. Ninguna
llave de API, ninguna llamada a Claude ni a Open-Meteo desde el navegador —
todo pasa por el backend.

## 7. Empaquetado

`frontend/Dockerfile` (multi-stage: `node:22` build → `nginx:1.27` sirve la
SPA). `frontend/nginx.conf` hace *fallback* a `index.html` y **proxy de `/api`**
al backend (incluido el SSE, con `proxy_buffering off`). En `docker-compose.yml`:
servicio `frontend` en el puerto 5173.

```bash
docker compose up -d      # db + backend + frontend  ->  http://localhost:5173
```

En desarrollo, Vite hace el proxy de `/api` a `127.0.0.1:8000` (`vite.config.ts`).

## 8. Rendimiento

- Code-splitting: `recharts` y `react` en *chunks* propios (cacheables por
  separado). App ≈ 29 KB, react ≈ 46 KB, recharts ≈ 553 KB (156 KB gzip).
- Assets con *hash* en el nombre → cacheados 1 año por Nginx.
- `staleTime` de 30 s en las queries para no recargar de más.

---

## Q-gate fase 8 → fase 9

✅ Dashboard responsive con las 7 secciones y las 8 visualizaciones · SSE en vivo
· TanStack Query · estados de carga/error/vacío · accesibilidad · sin lógica de
negocio ni secretos en el frontend · Docker (Nginx) · `npm run build` sin errores
de tipos.

**Siguiente (fase 9):** reentrenamiento — job que consolida los datos nuevos,
reentrena y **solo promociona** el modelo si mejora; robustez (logging, tests) y
`README` de reproducibilidad.
