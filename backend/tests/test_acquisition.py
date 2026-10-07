"""Pruebas de la capa de adquisición (fase 2).

Las pruebas de parsing usan un CSV de ejemplo (offline). Las que tocan la red
van marcadas con ``@pytest.mark.network`` y se pueden omitir:
    pytest -m "not network"
"""

from __future__ import annotations

import textwrap
from datetime import date

import pytest

from backend.ml.acquisition.metadata import build_metadata
from backend.ml.acquisition.open_meteo_archive import OpenMeteoArchiveClient
from backend.ml.config_loader import load_config

_SAMPLE_CSV = textwrap.dedent(
    """\
    latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation
    40.52724,-3.5593262,618.0,0,GMT,GMT

    time,temperature_2m (°C),relative_humidity_2m (%),wind_direction_10m (°)
    2010-01-01T00:00,3.1,71,270
    2010-01-01T01:00,2.6,74,268
    2010-01-01T02:00,,75,265
    """
)


def test_parse_csv_strips_units_and_parses_timestamp():
    df = OpenMeteoArchiveClient._parse_csv(_SAMPLE_CSV)
    assert list(df.columns) == [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_direction_10m",
    ]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["temperature_2m"].isna().sum() == 1
    assert df.attrs["grid_elevation_m"] == 618.0


def test_build_metadata_reports_range_and_missing(tmp_path):
    df = OpenMeteoArchiveClient._parse_csv(_SAMPLE_CSV)
    data_path = tmp_path / "sample.parquet"
    df.to_parquet(data_path, index=False)

    meta = build_metadata(
        data_path=data_path,
        df=df,
        timestamp_column="timestamp",
        source="test",
        source_url="http://example.test",
        license_name="CC BY 4.0",
    )
    assert meta["n_rows"] == 3
    assert meta["missing_values_per_column"]["temperature_2m"] == 1
    assert meta["expected_hourly_rows"] == 3
    assert len(meta["sha256"]) == 64


def test_config_matches_madrid_barajas():
    cfg = load_config()
    assert cfg.slug == "madrid_barajas"
    assert cfg.latitude == pytest.approx(40.4936)
    assert cfg.meteostat_station_id == "08221"
    assert "temperature_2m" in cfg.hourly_variables
    assert cfg.historical_start_date == date(2000, 1, 1)


@pytest.mark.network
def test_open_meteo_one_month_live():
    cfg = load_config()
    client = OpenMeteoArchiveClient(
        base_url=cfg.archive_base_url,
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        variables=["temperature_2m", "relative_humidity_2m"],
        cache_dir=None,
    )
    df = client.fetch_hourly(date(2015, 6, 1), date(2015, 6, 30))
    assert len(df) == 30 * 24
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()
