"""Pruebas de la partición cronológica (fase 3, anti data leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ml.preparation.split import chronological_split


def _frame(n: int = 10_000) -> pd.DataFrame:
    ts = pd.date_range("2005-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "x": np.arange(n)})


def test_splits_are_ordered_in_time_with_a_gap():
    r = chronological_split(_frame(), gap_hours=24)
    t_end = pd.Timestamp(r.boundaries["train"]["end"])
    v_start = pd.Timestamp(r.boundaries["valid"]["start"])
    v_end = pd.Timestamp(r.boundaries["valid"]["end"])
    s_start = pd.Timestamp(r.boundaries["test"]["start"])

    assert t_end < v_start < v_end < s_start
    assert (v_start - t_end) >= pd.Timedelta(hours=24)
    assert (s_start - v_end) >= pd.Timedelta(hours=24)


def test_no_timestamp_appears_in_two_splits():
    r = chronological_split(_frame())
    ts_train = set(r.train["timestamp"])
    ts_valid = set(r.valid["timestamp"])
    ts_test = set(r.test["timestamp"])
    assert ts_train.isdisjoint(ts_valid)
    assert ts_valid.isdisjoint(ts_test)
    assert ts_train.isdisjoint(ts_test)


def test_fractions_are_approximately_respected():
    r = chronological_split(_frame(100_000), fractions=(0.70, 0.15, 0.15))
    total = r.boundaries["total_rows"]
    assert abs(len(r.train) / total - 0.70) < 0.01
    assert abs(len(r.valid) / total - 0.15) < 0.01
    assert abs(len(r.test) / total - 0.15) < 0.01


def test_train_is_the_oldest_data():
    r = chronological_split(_frame())
    assert r.train["x"].max() < r.valid["x"].min()
    assert r.valid["x"].max() < r.test["x"].min()
