"""Shared pytest fixtures for the Greek DAM forecasting test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.Series:
    """
    Deterministic synthetic hourly prices for testing.

    90 days of clean hourly data with a daily + weekly pattern, no missing values,
    no DST transition. Intentionally small and predictable.
    """
    start = pd.Timestamp("2024-01-01", tz="Europe/Athens")
    end = pd.Timestamp("2024-04-01", tz="Europe/Athens")
    idx = pd.date_range(start=start, end=end, freq="h", inclusive="left")

    rng = np.random.default_rng(seed=42)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    base = 100.0
    daily = 30 * np.sin((hour - 7) * np.pi / 12)
    weekly = np.where(dow >= 5, -15.0, 0.0)
    noise = rng.normal(0, 5, len(idx))

    values = base + daily + weekly + noise
    return pd.Series(values, index=idx, name="price_eur_mwh")


@pytest.fixture
def short_prices() -> pd.Series:
    """A shorter price series — only 30 days. For tests that just need *some* data."""
    start = pd.Timestamp("2024-01-01", tz="Europe/Athens")
    end = pd.Timestamp("2024-01-31", tz="Europe/Athens")
    idx = pd.date_range(start=start, end=end, freq="h", inclusive="left")

    rng = np.random.default_rng(seed=42)
    values = 100 + 20 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 3, len(idx))
    return pd.Series(values, index=idx, name="price_eur_mwh")


@pytest.fixture
def dst_spanning_prices() -> pd.Series:
    """Prices spanning the spring-forward DST transition (March 31, 2024)."""
    start = pd.Timestamp("2024-03-15", tz="Europe/Athens")
    end = pd.Timestamp("2024-04-15", tz="Europe/Athens")
    idx = pd.date_range(start=start, end=end, freq="h", inclusive="left")

    rng = np.random.default_rng(seed=42)
    values = 100 + 20 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 3, len(idx))
    return pd.Series(values, index=idx, name="price_eur_mwh")