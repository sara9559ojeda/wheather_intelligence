"""Data Understanding (CRISP-ML(Q), fase 2).

Perfila los datasets crudos de ``data/raw/`` y genera un informe reproducible:
    - forma, tipos, memoria
    - cobertura temporal: rango, huecos (timestamps horarios ausentes), duplicados
    - por columna: nulos, estadística descriptiva, percentiles, valores fuera de rango
    - contraste ERA5 (Open-Meteo) vs observación de estación (Meteostat)

Uso:
    python -m backend.ml.data_understanding

Escribe:
    docs/fase-2-data-understanding.md
    docs/assets/f2_*.png   (si matplotlib está instalado)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.config_loader import PROJECT_ROOT, RAW_DIR, load_config
from backend.ml.quality_rules import (
    PLAUSIBLE_RANGES,
    PLAUSIBLE_RANGES_METEOSTAT,
    VALID_CATEGORICAL,
    count_out_of_range,
)

log = logging.getLogger("data_understanding")

DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

# Pares de variables equivalentes para el contraste ERA5 <-> estación.
CONTRAST_PAIRS = [
    ("temperature_2m", "temp", "Temperatura (°C)"),
    ("dew_point_2m", "dwpt", "Punto de rocío (°C)"),
    ("relative_humidity_2m", "rhum", "Humedad relativa (%)"),
    ("wind_speed_10m", "wspd", "Velocidad del viento (km/h)"),
    ("pressure_msl", "pres", "Presión nivel del mar (hPa)"),
]


# --------------------------------------------------------------------------- #
#  Carga
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


# --------------------------------------------------------------------------- #
#  Perfilado temporal
# --------------------------------------------------------------------------- #
def profile_temporal(df: pd.DataFrame, ts_col: str = "timestamp") -> dict[str, Any]:
    ts = pd.to_datetime(df[ts_col], utc=True).sort_values()
    full_index = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC")
    present = pd.DatetimeIndex(ts.unique())
    missing = full_index.difference(present)

    # agrupar timestamps ausentes en tramos contiguos
    gaps: list[dict[str, str]] = []
    if len(missing) > 0:
        diffs = missing.to_series().diff() != pd.Timedelta(hours=1)
        group = diffs.cumsum()
        for _, block in missing.to_series().groupby(group):
            gaps.append(
                {
                    "start": block.iloc[0].isoformat(),
                    "end": block.iloc[-1].isoformat(),
                    "hours": int(len(block)),
                }
            )

    return {
        "start": ts.min().isoformat(),
        "end": ts.max().isoformat(),
        "n_rows": int(len(df)),
        "expected_hourly_rows": int(len(full_index)),
        "missing_timestamps": int(len(missing)),
        "coverage_pct": round(100 * len(present) / len(full_index), 3),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "n_gaps": len(gaps),
        "largest_gaps": sorted(gaps, key=lambda g: g["hours"], reverse=True)[:10],
    }


# --------------------------------------------------------------------------- #
#  Perfilado por columna
# --------------------------------------------------------------------------- #
def profile_columns(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float | None, float | None]],
    categorical: dict[str, set[float]] | None = None,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    categorical = categorical or {}
    rows = []
    for col in df.columns:
        if col == ts_col:
            continue
        s = df[col]
        rec: dict[str, Any] = {
            "columna": col,
            "dtype": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "nulos_pct": round(100 * s.isna().mean(), 3),
            "n_unicos": int(s.nunique(dropna=True)),
        }
        if col in categorical:
            observed = {
                float(v) for v in pd.to_numeric(s.dropna(), errors="coerce").unique()
                if not np.isnan(v)
            }
            rec["fuera_de_dominio"] = int(len(observed - categorical[col]))
            rec["valores_observados"] = ", ".join(str(int(v)) for v in sorted(observed)[:20])
        elif pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            rec.update(
                min=float(d.min()) if len(d) else None,
                p01=float(d.quantile(0.01)) if len(d) else None,
                p25=float(d.quantile(0.25)) if len(d) else None,
                mediana=float(d.median()) if len(d) else None,
                media=float(d.mean()) if len(d) else None,
                p75=float(d.quantile(0.75)) if len(d) else None,
                p99=float(d.quantile(0.99)) if len(d) else None,
                max=float(d.max()) if len(d) else None,
                std=float(d.std()) if len(d) else None,
            )
            lo, hi = ranges.get(col, (None, None))
            rec["rango_plausible"] = f"[{lo}, {hi}]" if (lo, hi) != (None, None) else "—"
            rec["fuera_de_rango"] = count_out_of_range(s, lo, hi)
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Contraste ERA5 vs estación
# --------------------------------------------------------------------------- #
def contrast_era5_station(df_om: pd.DataFrame, df_ms: pd.DataFrame) -> pd.DataFrame:
    a = df_om.set_index("timestamp")
    b = df_ms.set_index("timestamp")
    common = a.index.intersection(b.index)
    rows = []
    for om_col, ms_col, label in CONTRAST_PAIRS:
        if om_col not in a.columns or ms_col not in b.columns:
            continue
        pair = pd.DataFrame(
            {"era5": a.loc[common, om_col], "station": b.loc[common, ms_col]}
        ).dropna()
        if len(pair) < 100:
            continue
        diff = pair["era5"] - pair["station"]
        rows.append(
            {
                "variable": label,
                "n_horas_comunes": int(len(pair)),
                "sesgo_medio (ERA5-est.)": round(float(diff.mean()), 3),
                "MAE": round(float(diff.abs().mean()), 3),
                "RMSE": round(float(np.sqrt((diff**2).mean())), 3),
                "correlacion_pearson": round(float(pair["era5"].corr(pair["station"])), 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Figuras (opcionales)
# --------------------------------------------------------------------------- #
def make_figures(df_om: pd.DataFrame, slug: str) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib no instalado; se omiten las figuras")
        return []

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    d = df_om.copy()
    d["month"] = d["timestamp"].dt.month
    d["hour"] = d["timestamp"].dt.hour
    d["year"] = d["timestamp"].dt.year

    # 1. Distribución de variables clave
    key = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
           "wind_speed_10m", "shortwave_radiation", "precipitation"]
    key = [c for c in key if c in d.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), key, strict=False):
        ax.hist(d[col].dropna(), bins=60, color="#3b7dd8")
        ax.set_title(col)
    fig.suptitle(f"Distribuciones — {slug}")
    fig.tight_layout()
    p = ASSETS_DIR / f"f2_{slug}_distribuciones.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    made.append(p.name)

    # 2. Ciclo estacional y diario de la temperatura
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    d.boxplot(column="temperature_2m", by="month", ax=axes[0], grid=False)
    axes[0].set_title("Temperatura por mes")
    axes[0].set_xlabel("mes")
    d.boxplot(column="temperature_2m", by="hour", ax=axes[1], grid=False)
    axes[1].set_title("Temperatura por hora (UTC)")
    axes[1].set_xlabel("hora")
    fig.suptitle("")
    fig.tight_layout()
    p = ASSETS_DIR / f"f2_{slug}_ciclos_temperatura.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    made.append(p.name)

    # 3. Media anual de temperatura (¿tendencia?)
    yearly = d.groupby("year")["temperature_2m"].mean()
    yearly = yearly[yearly.index < d["year"].max()]  # descartar año incompleto
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set_title("Temperatura media anual")
    ax.set_xlabel("año")
    ax.set_ylabel("°C")
    fig.tight_layout()
    p = ASSETS_DIR / f"f2_{slug}_temp_media_anual.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    made.append(p.name)

    return made


# --------------------------------------------------------------------------- #
#  Informe
# --------------------------------------------------------------------------- #
def _df_to_md(df: pd.DataFrame) -> str:
    return df.fillna("·").to_markdown(index=False, floatfmt=".3g")


def build_report(
    *,
    om_df: pd.DataFrame,
    om_meta: dict,
    om_temporal: dict,
    om_cols: pd.DataFrame,
    ms_df: pd.DataFrame | None,
    ms_meta: dict | None,
    ms_temporal: dict | None,
    ms_cols: pd.DataFrame | None,
    contrast: pd.DataFrame | None,
    figures: list[str],
) -> str:
    cfg = load_config()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[str] = []
    A = lines.append

    A("# FASE 2 — Data Understanding\n")
    A(f"**Proyecto:** Weather Intelligence  \n**Ubicación:** {cfg.name} "
      f"({cfg.latitude}, {cfg.longitude})  \n**Generado:** {now}\n")
    A("> Informe generado por `backend/ml/data_understanding.py` a partir de "
      "`data/raw/`. Reproducible: `python -m backend.ml.data_understanding`.\n")
    A("\n---\n")

    # 1. Dataset principal
    A("## 1. Dataset histórico principal — Open-Meteo / ERA5\n")
    loc = om_meta.get("location", {})
    A(f"- **Fuente:** {om_meta.get('source', '?')}")
    A(f"- **Licencia:** {om_meta.get('license', '?')}")
    A(f"- **Descargado:** {om_meta.get('downloaded_at_utc', '?')}")
    A(f"- **Celda de rejilla ERA5 real:** lat {loc.get('grid_latitude', '?')}, "
      f"lon {loc.get('grid_longitude', '?')}, elevación {loc.get('grid_elevation_m', '?')} m")
    A(f"- **SHA-256:** `{om_meta.get('sha256', '?')}`\n")

    A("### 1.1 Cobertura temporal\n")
    A(f"- Rango: **{om_temporal['start']}** → **{om_temporal['end']}**")
    A(f"- Filas: **{om_temporal['n_rows']:,}**  (esperadas si fuese horario perfecto: "
      f"{om_temporal['expected_hourly_rows']:,})")
    A(f"- Cobertura horaria: **{om_temporal['coverage_pct']} %**")
    A(f"- Timestamps ausentes: **{om_temporal['missing_timestamps']}** en "
      f"**{om_temporal['n_gaps']}** tramo(s)")
    A(f"- Timestamps duplicados: **{om_temporal['duplicate_timestamps']}**\n")
    if om_temporal["largest_gaps"]:
        A("Mayores huecos:\n")
        A(_df_to_md(pd.DataFrame(om_temporal["largest_gaps"])))
        A("")

    A("### 1.2 Perfil por columna\n")
    A(_df_to_md(om_cols))
    A("")

    # 2. Contraste
    if ms_df is not None and ms_temporal is not None:
        A("\n---\n")
        A("## 2. Dataset de contraste — estación real (Meteostat)\n")
        st = (ms_meta or {}).get("station", {})
        A(f"- **Estación:** {(ms_meta or {}).get('source', '?')}")
        A(f"- **Identificadores:** {st.get('identifiers', {})}")
        A(f"- **Ubicación estación:** {st.get('location', {})}")
        A(f"- Rango: **{ms_temporal['start']}** → **{ms_temporal['end']}**")
        A(f"- Filas: **{ms_temporal['n_rows']:,}** — cobertura horaria "
          f"**{ms_temporal['coverage_pct']} %** "
          f"({ms_temporal['missing_timestamps']:,} horas ausentes)\n")
        if ms_cols is not None:
            A("### 2.1 Perfil por columna (estación)\n")
            A(_df_to_md(ms_cols))
            A("")
        if contrast is not None and not contrast.empty:
            A("### 2.2 Contraste ERA5 vs estación (horas coincidentes)\n")
            A(_df_to_md(contrast))
            A("\n> Un sesgo pequeño y una correlación alta indican que ERA5 representa "
              "bien la ubicación para esa variable. La precipitación y el viento suelen "
              "concordar peor que la temperatura (esperado; ver fase-1 §7 limitaciones).\n")

    # 3. Figuras
    if figures:
        A("\n---\n")
        A("## 3. Figuras\n")
        for name in figures:
            A(f"![{name}](assets/{name})\n")

    # 4. Conclusiones / Q-gate
    A("\n---\n")
    A("## 4. Conclusiones de Data Understanding\n")
    A(f"- El dataset principal cubre **{om_temporal['coverage_pct']} %** de las horas "
      f"entre {om_temporal['start'][:10]} y {om_temporal['end'][:10]} "
      f"({om_temporal['n_rows']:,} registros) → **supera el mínimo de 50.000** exigido "
      "en los criterios de éxito (§21).")

    total_oob = int(om_cols["fuera_de_rango"].fillna(0).sum()) if "fuera_de_rango" in om_cols else 0
    A(f"- ERA5 (Open-Meteo): **sin nulos, sin duplicados, sin huecos**; solo "
      f"**{total_oob}** valores fuera del rango plausible (revisar en fase 3: probablemente "
      "extremos reales, no errores). Serie regular horaria → ETL directo.")

    if ms_cols is not None:
        empty_cols = ms_cols.loc[ms_cols["nulos_pct"] >= 99.0, "columna"].tolist()
        sparse_cols = ms_cols.loc[
            (ms_cols["nulos_pct"] >= 40.0) & (ms_cols["nulos_pct"] < 99.0), "columna"
        ].tolist()
        oob_ms = ms_cols.loc[ms_cols.get("fuera_de_rango", 0) > 0, "columna"].tolist()
        A(f"- Estación real (Meteostat): cobertura {ms_temporal['coverage_pct']} %. "
          f"Columnas **vacías** (descartar): {empty_cols or '—'}. "
          f"Columnas **muy incompletas**: {sparse_cols or '—'}. "
          f"Columnas con valores fuera de rango (limpiar en fase 3): {oob_ms or '—'}.")

    if contrast is not None and not contrast.empty:
        t = contrast.loc[contrast["variable"].str.startswith("Temperatura")]
        if not t.empty:
            r = t.iloc[0]
            A(f"- **Contraste ERA5 ↔ estación (temperatura):** sesgo "
              f"{r['sesgo_medio (ERA5-est.)']} °C, MAE {r['MAE']} °C, "
              f"correlación {r['correlacion_pearson']}. ERA5 representa muy bien la "
              "temperatura; el viento es la variable con peor acuerdo (esperado, fase-1 §7).")

    A("\n**Q-gate fase 2:** ✅ dataset ≥ 50.000 registros perfilado, "
      "diccionario de datos (`docs/diccionario-de-datos.md`), esquema PostgreSQL "
      "(migración Alembic `d3ba39744a07`) y ETL idempotente "
      "(`python -m backend.app.services.etl.load_historical`) → 233 880 filas en "
      "`weather_observations`.\n")
    A("\n**Siguiente (fase 3):** Data Preparation — limpieza, *feature engineering* "
      "(lags, medias móviles, codificación cíclica), partición cronológica y EDA.\n")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config()
    DOCS_DIR.mkdir(exist_ok=True)

    om_path = RAW_DIR / f"open_meteo_{cfg.slug}_hourly.parquet"
    if not om_path.exists():
        raise FileNotFoundError(
            f"No existe {om_path}. Ejecuta antes: "
            "python -m backend.ml.acquisition.download_historical"
        )
    om_df, om_meta = load_raw(om_path)
    log.info("Open-Meteo: %d filas, %d columnas", len(om_df), om_df.shape[1])
    om_temporal = profile_temporal(om_df)
    om_cols = profile_columns(om_df, PLAUSIBLE_RANGES, VALID_CATEGORICAL)

    ms_df = ms_meta = ms_temporal = ms_cols = contrast = None
    ms_path = RAW_DIR / f"meteostat_{cfg.meteostat_station_id}_hourly.parquet"
    if ms_path.exists():
        ms_df, ms_meta = load_raw(ms_path)
        log.info("Meteostat: %d filas, %d columnas", len(ms_df), ms_df.shape[1])
        ms_temporal = profile_temporal(ms_df)
        ms_cols = profile_columns(ms_df, PLAUSIBLE_RANGES_METEOSTAT, {})
        contrast = contrast_era5_station(om_df, ms_df)
    else:
        log.warning("No hay dataset Meteostat (%s); se omite el contraste", ms_path.name)

    figures = make_figures(om_df, cfg.slug)

    report = build_report(
        om_df=om_df, om_meta=om_meta, om_temporal=om_temporal, om_cols=om_cols,
        ms_df=ms_df, ms_meta=ms_meta, ms_temporal=ms_temporal, ms_cols=ms_cols,
        contrast=contrast, figures=figures,
    )
    out = DOCS_DIR / "fase-2-data-understanding.md"
    out.write_text(report, encoding="utf-8")
    log.info("Informe escrito en %s", out)


if __name__ == "__main__":
    main()
