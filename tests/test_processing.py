"""Tests for src/processing.py."""

import numpy as np
import pandas as pd
import pytest

from src.processing import ensure_hourly, ensure_hourly_df


def test_ensure_hourly_passes_through_hourly_data():
    idx = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    s = pd.Series(np.arange(10.0), index=idx, name="x")
    out = ensure_hourly(s)
    pd.testing.assert_series_equal(out, s)


def test_ensure_hourly_aggregates_15min_to_hourly():
    idx = pd.date_range("2025-01-01 00:00", periods=8, freq="15min", tz="UTC")
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], index=idx, name="x")
    out = ensure_hourly(s)
    assert len(out) == 2
    assert out.iloc[0] == 25.0  # mean(10,20,30,40)
    assert out.iloc[1] == 65.0  # mean(50,60,70,80)


def test_ensure_hourly_handles_short_series():
    s_empty = pd.Series([], dtype=float)
    assert len(ensure_hourly(s_empty)) == 0

    idx = pd.DatetimeIndex(["2025-01-01"], tz="UTC")
    s_one = pd.Series([42.0], index=idx)
    assert len(ensure_hourly(s_one)) == 1


def test_ensure_hourly_df_handles_mixed_cadence():
    """Multi-column DataFrame at 15-min cadence aggregates to hourly."""
    idx = pd.date_range("2025-01-01 00:00", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "solar": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "wind":  [10.0,   20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0],
    }, index=idx)
    out = ensure_hourly_df(df)
    assert len(out) == 2
    assert out["solar"].iloc[0] == 250.0   # mean(100,200,300,400)
    assert out["wind"].iloc[1] == 65.0     # mean(50,60,70,80)


def test_ensure_hourly_partial_quarter_at_boundary():
    """If only some hours have 15-min data, mixed input should aggregate consistently."""
    # First hour: 1 hourly point. Second hour: 4 quarters.
    idx = pd.DatetimeIndex([
        "2025-01-01 00:00",
        "2025-01-01 01:00",
        "2025-01-01 01:15",
        "2025-01-01 01:30",
        "2025-01-01 01:45",
    ], tz="UTC")
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx, name="x")
    out = ensure_hourly(s)
    # Two hours: hour 00 = 10 (single point), hour 01 = mean(20,30,40,50) = 35
    assert len(out) == 2
    assert out.iloc[0] == 10.0
    assert out.iloc[1] == 35.0