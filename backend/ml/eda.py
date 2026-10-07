"""Análisis Exploratorio de Datos (EDA) — fase 3.

Genera ``docs/fase-3-eda.md`` + figuras en ``docs/assets/``.

Se ejecuta sobre **train + valid** (2000–2022). El tramo de **test** NO se toca
en el EDA: se reserva para la evaluación final (fase 5), y mirarlo ahora podría
sesgar decisiones de modelado.

    python -m backend.ml.eda
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from backend.ml.config_loader import PROCESSED_DIR, PROJECT_ROOT, load_config

log = logging.getLogger("eda")

DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

KEY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
    "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "shortwave_radiation",
    "precipitation", "et0_fao_evapotranspiration",
]
BIVARIATE_VS_TEMP = [
    ("relative_humidity_2m", "Humedad relativa (%)"),
    ("surface_pressure", "Presión superficie (hPa)"),
    ("wind_speed_10m", "Viento (km/h)"),
    ("shortwave_radiation", "Radiación solar (W/m²)"),
    ("cloud_cover", "Nubosidad (%)"),
    ("dew_point_2m", "Punto de rocío (°C)"),
]


def _load_train_valid() -> pd.DataFrame:
    parts = []
    for name in ("train", "valid"):
        p = PROCESSED_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Falta {p}. Ejecuta antes: python -m backend.ml.preparation.build_dataset"
            )
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
def univariate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in KEY_VARS:
        s = df[col].dropna()
        rows.append({
            "variable": col,
            "media": s.mean(), "mediana": s.median(),
            "std": s.std(), "var": s.var(),
            "min": s.min(), "p05": s.quantile(0.05), "p25": s.quantile(0.25),
            "p75": s.quantile(0.75), "p95": s.quantile(0.95), "max": s.max(),
        })
    return pd.DataFrame(rows)


def bivariate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in BIVARIATE_VS_TEMP:
        pair = df[["temperature_2m", col]].dropna()
        rows.append({
            "relación": f"temperature_2m vs {col}",
            "descripción": label,
            "pearson": pair["temperature_2m"].corr(pair[col], method="pearson"),
            "spearman": pair["temperature_2m"].corr(pair[col], method="spearman"),
        })
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_VARS if c in df.columns]
    return df[cols].corr(method="pearson")


# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, slug: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib no instalado; se omiten las figuras")
        return []

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["month"] = d["timestamp"].dt.month
    d["dayofweek"] = d["timestamp"].dt.dayofweek

    # 1. histogramas
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), KEY_VARS[:6], strict=False):
        ax.hist(d[col].dropna(), bins=60, color="#3b7dd8")
        ax.set_title(col)
    fig.suptitle(f"Distribuciones univariadas — {slug} (train+valid)")
    fig.tight_layout()
    p = ASSETS_DIR / f"f3_{slug}_histogramas.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 2. bivariado: temp vs otras (hexbin por densidad)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, label) in zip(axes.ravel(), BIVARIATE_VS_TEMP, strict=False):
        pair = d[["temperature_2m", col]].dropna()
        ax.hexbin(pair[col], pair["temperature_2m"], gridsize=45, cmap="Blues", mincnt=1)
        ax.set_xlabel(label); ax.set_ylabel("Temperatura (°C)")
        r = pair["temperature_2m"].corr(pair[col])
        ax.set_title(f"r = {r:.2f}")
    fig.suptitle("Temperatura frente a otras variables")
    fig.tight_layout()
    p = ASSETS_DIR / f"f3_{slug}_bivariado.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 3. temporal
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    d.boxplot(column="temperature_2m", by="hour", ax=axes[0], grid=False)
    axes[0].set_title("por hora (UTC)"); axes[0].set_xlabel("hora")
    d.boxplot(column="temperature_2m", by="dayofweek", ax=axes[1], grid=False)
    axes[1].set_title("por día de semana"); axes[1].set_xlabel("0=lun … 6=dom")
    d.boxplot(column="temperature_2m", by="month", ax=axes[2], grid=False)
    axes[2].set_title("por mes"); axes[2].set_xlabel("mes")
    fig.suptitle("Temperatura: patrones temporales")
    fig.tight_layout()
    p = ASSETS_DIR / f"f3_{slug}_temporal.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    # 4. matriz de correlación
    corr = correlation_matrix(d)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlación (Pearson)")
    fig.tight_layout()
    p = ASSETS_DIR / f"f3_{slug}_correlaciones.png"
    fig.savefig(p, dpi=110); plt.close(fig); made.append(p.name)

    return made


# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3g")


def build_report(df: pd.DataFrame, figures: list[str]) -> str:
    cfg = load_config()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    uni = univariate_table(df)
    biv = bivariate_table(df)
    corr = correlation_matrix(df)
    temp_corr = corr["temperature_2m"].drop("temperature_2m").sort_values(key=abs, ascending=False)

    L: list[str] = []
    A = L.append
    A("# FASE 3 — Análisis Exploratorio de Datos (EDA)\n")
    A(f"**Ubicación:** {cfg.name}  ·  **Datos:** train + valid "
      f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()}, "
      f"{len(df):,} filas)  ·  **Generado:** {now}\n")
    A("> El tramo de **test** se reserva para la fase 5 y no se analiza aquí "
      "(evita sesgar decisiones de modelado). Reproducible: `python -m backend.ml.eda`.\n")
    A("\n---\n")

    A("## 1. Análisis univariado\n")
    A(_md(uni))
    A("\n**Lectura:**")
    A("- `temperature_2m`: distribución ancha (std ≈ 9 °C) por el fuerte ciclo anual; "
      "ligeramente asimétrica hacia valores altos en verano.")
    A("- `precipitation` y `shortwave_radiation`: fuertemente asimétricas a la derecha "
      "(muchos ceros: horas sin lluvia / de noche). No se deben normalizar de forma ingenua.")
    A("- `surface_pressure`: casi simétrica y estrecha (std ≈ 6 hPa).\n")

    A("\n---\n")
    A("## 2. Análisis bivariado (frente a la temperatura)\n")
    A(_md(biv))
    A("\n**Lectura:**")
    A("- **Humedad relativa:** correlación negativa moderada — el aire más cálido "
      "admite más vapor, así que a igual humedad absoluta la relativa baja.")
    A("- **Radiación solar:** correlación positiva — más radiación calienta la superficie, "
      "pero la relación está mediada por la hora del día y la estación.")
    A("- **Punto de rocío:** correlación positiva fuerte (comparten la componente estacional).")
    A("- La relación con la **presión** y el **viento** es débil y no lineal.\n")

    A("\n---\n")
    A("## 3. Análisis temporal\n")
    hourly = df.assign(h=df["timestamp"].dt.hour).groupby("h")["temperature_2m"].mean()
    monthly = df.assign(m=df["timestamp"].dt.month).groupby("m")["temperature_2m"].mean()
    A(f"- **Ciclo diario:** mínimo hacia las {hourly.idxmin():02d}:00 UTC "
      f"({hourly.min():.1f} °C), máximo hacia las {hourly.idxmax():02d}:00 UTC "
      f"({hourly.max():.1f} °C). Amplitud media ≈ {hourly.max() - hourly.min():.1f} °C.")
    A(f"- **Ciclo anual:** mes más frío = {monthly.idxmin()} ({monthly.min():.1f} °C), "
      f"más cálido = {monthly.idxmax()} ({monthly.max():.1f} °C).")
    A("- **Día de la semana:** sin patrón relevante (el clima no sabe si es lunes) — "
      "sirve como comprobación de que no hay artefactos de muestreo.\n")

    A("\n---\n")
    A("## 4. Correlaciones\n")
    A("Matriz de correlación de Pearson entre variables crudas:\n")
    A(_md(corr.reset_index().rename(columns={"index": ""})))
    A("\n**Variables más correlacionadas con la temperatura** (|Pearson|):\n")
    A(_md(temp_corr.round(3).reset_index().rename(
        columns={"index": "variable", "temperature_2m": "pearson"})))
    A("\n> ⚠️ **Correlación no implica causalidad.** Muchas de estas relaciones están "
      "confundidas por variables comunes (hora del día, estación del año): por ejemplo, "
      "radiación y temperatura suben juntas porque ambas dependen de la posición solar, "
      "no porque una cause directamente a la otra en estos datos.\n")

    if figures:
        A("\n---\n")
        A("## 5. Figuras\n")
        for name in figures:
            A(f"![{name}](assets/{name})\n")

    A("\n---\n")
    A("## 6. Conclusiones para el modelado (fase 5)\n")
    A("- La temperatura está dominada por dos ciclos (diario y anual) → las features "
      "cíclicas (`hour_sin/cos`, `doy_sin/cos`) y los lags de 24 h deberían ser muy "
      "informativos.")
    A("- La persistencia (`temperature_2m_lag_1h`) será un *baseline* fuerte a horizontes "
      "cortos; el reto está en 6–24 h.")
    A("- Variables muy asimétricas (precipitación, radiación) conviene tratarlas con "
      "transformaciones suaves o usarlas vía agregados (`precipitation_roll_sum_*`).")
    A("- No hay señal por día de semana → no se incluye `dayofweek` como predictor fuerte "
      "(se deja por control).\n")
    return "\n".join(L)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config()
    df = _load_train_valid()
    log.info("EDA sobre %d filas (%s → %s)", len(df),
             df["timestamp"].min().date(), df["timestamp"].max().date())
    figures = make_figures(df, cfg.slug)
    report = build_report(df, figures)
    out = DOCS_DIR / "fase-3-eda.md"
    out.write_text(report, encoding="utf-8")
    log.info("Informe escrito en %s", out)


if __name__ == "__main__":
    main()
