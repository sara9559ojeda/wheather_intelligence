"""F2 — patrones meteorológicos: etiquetas derivadas de los perfiles, mismo preprocesamiento,
episodios y transiciones sobre datos ordenados en el tiempo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from backend.app.db.models import ClusterModel, Location
from backend.app.db.session import SessionLocal
from backend.app.services.analysis import regimes as rg
from backend.ml.mining.clustering import CLUSTER_FEATURES, build_matrix

NEUTRAL_RANKS = {f: 0.5 for f in CLUSTER_FEATURES}
NEUTRAL_Z = {f: 0.0 for f in CLUSTER_FEATURES}


def _medians(**over) -> dict:
    base = {f: 10.0 for f in CLUSTER_FEATURES}
    base["shortwave_radiation"] = 200.0
    base.update(over)
    return base


# ---------------------------------------------------------------- etiquetas
def test_labels_come_from_statistics_not_from_a_fixed_list():
    ranks = {**NEUTRAL_RANKS, "relative_humidity_2m": 0.9, "cloud_cover": 0.85}
    z = {**NEUTRAL_Z, "relative_humidity_2m": 1.2, "cloud_cover": 1.0}
    words = [w for w, _ in rg._descriptors(_medians(), 2.0, z, ranks)]
    assert "húmedo" in words and "nublado" in words
    assert "seco" not in words and "lluvioso" not in words


def test_no_descriptor_is_invented_for_average_regimes():
    assert rg._descriptors(_medians(), 2.0, NEUTRAL_Z, NEUTRAL_RANKS) == []


def test_night_and_rain_descriptors_follow_the_data():
    words = [w for w, _ in rg._descriptors(
        _medians(shortwave_radiation=0.0), 40.0, NEUTRAL_Z, NEUTRAL_RANKS)]
    assert "nocturno" in words and "lluvioso" in words
    dry_day = [w for w, _ in rg._descriptors(
        _medians(shortwave_radiation=300.0), 0.5, NEUTRAL_Z,
        {**NEUTRAL_RANKS, "shortwave_radiation": 0.9})]
    assert "soleado" in dry_day and "nocturno" not in dry_day


def test_labels_inside_a_city_are_unique_even_with_identical_descriptors():
    used: set[str] = set()
    descs = [("húmedo", 1.0), ("nublado", 0.5)]
    labels = []
    for _ in range(4):
        label = rg._unique_label("fresco", descs, used)
        used.add(label)
        labels.append(label)
    assert len(set(labels)) == 4


def test_compose_reads_as_a_phrase():
    assert rg._compose("cálido", []) == "Cálido"
    assert rg._compose("cálido", ["seco"]) == "Cálido y seco"
    assert rg._compose("cálido", ["seco", "soleado"]) == "Cálido, seco y soleado"


# ---------------------------------------------------------------- preprocesamiento
def test_assign_frame_uses_the_training_scaler_and_matrix():
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame({f: rng.normal(10, 3, n) for f in CLUSTER_FEATURES})
    frame["precipitation"] = np.where(rng.random(n) < 0.2, rng.exponential(2, n), 0.0)
    train = build_matrix(frame.iloc[:400])
    scaler = StandardScaler().fit(train)
    km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(scaler.transform(train))
    art = {"kmeans": km, "scaler": scaler}

    out = rg.assign_frame(art, frame.iloc[400:])
    x = build_matrix(frame.iloc[400:])              # log1p de la precipitación incluido
    expected = km.predict(scaler.transform(x))
    assert (out["regime"].to_numpy() == expected).all()
    d = km.transform(scaler.transform(x))
    assert np.allclose(out["distance"].to_numpy(), d.min(axis=1))
    assert (out["second_distance"] >= out["distance"]).all()


def test_assign_frame_skips_rows_with_missing_values():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({f: rng.normal(10, 3, 100) for f in CLUSTER_FEATURES})
    frame["precipitation"] = frame["precipitation"].abs()
    art_x = build_matrix(frame)
    scaler = StandardScaler().fit(art_x)
    km = KMeans(n_clusters=2, n_init=2, random_state=0).fit(scaler.transform(art_x))
    broken = frame.copy()
    broken.loc[broken.index[:5], "cloud_cover"] = np.nan
    out = rg.assign_frame({"kmeans": km, "scaler": scaler}, broken)
    assert len(out) == 95


# ---------------------------------------------------------------- episodios y transiciones
def _series(clusters: list[int], gaps_after: tuple[int, ...] = ()) -> pd.DataFrame:
    t = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i, c in enumerate(clusters):
        rows.append({"observed_at": t, "cluster": c})
        t += pd.Timedelta(hours=3 if i in gaps_after else 1)
    return pd.DataFrame(rows)


def test_runs_group_consecutive_hours_and_split_on_gaps():
    runs = rg._runs(_series([0, 0, 0, 1, 1, 0, 0], gaps_after=(5,)))
    assert runs["cluster"].tolist() == [0, 1, 0, 0]          # el hueco parte el último episodio
    assert runs["hours"].tolist() == [3, 2, 1, 1]


def test_transitions_ignore_self_loops_and_gaps():
    # 0→1 contiguo, 1→0 contiguo, 0→1 con hueco (no cuenta)
    runs = rg._runs(_series([0, 0, 1, 1, 0, 1], gaps_after=(4,)))
    tr = rg._transitions(runs)
    assert tr["n_pairs"] == 2
    assert tr["from"]["0"]["to"]["1"]["count"] == 1
    assert tr["from"]["1"]["to"]["0"]["count"] == 1
    assert all(abs(sum(v["probability"] for v in row["to"].values()) - 1) < 1e-3
               for row in tr["from"].values())


# ---------------------------------------------------------------- perfiles reales persistidos
@pytest.mark.db
def test_persisted_profiles_are_consistent_with_their_statistics():
    with SessionLocal() as s:
        rows = s.execute(
            select(ClusterModel, Location.slug)
            .join(Location, Location.id == ClusterModel.location_id)
            .where(ClusterModel.profile.is_not(None))
        ).all()
        if not rows:
            pytest.skip("Ningún modelo de clustering con perfil calculado")
        for cm, slug in rows:
            regs = cm.profile["regimes"]
            labels = [r["label"] for r in regs]
            assert len(labels) == len(set(labels)), f"{slug}: etiquetas repetidas"
            assert abs(sum(r["share_pct"] for r in regs) - 100) < 0.5, slug
            for r in regs:
                med = r["medians"]
                if "nocturno" in r["label"].lower():
                    assert med["shortwave_radiation"] < 5, (slug, r["label"])
                if "soleado" in r["label"].lower():
                    assert med["shortwave_radiation"] >= 5, (slug, r["label"])
                if "lluvioso" in r["label"].lower():
                    assert r["wet_share_pct"] >= 25, (slug, r["label"])
            assert cm.cluster_labels is not None
            assert {int(k) for k in cm.cluster_labels} == {r["regime_id"] for r in regs}
