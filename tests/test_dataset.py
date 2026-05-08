"""Tests for src/dataset.py."""

import numpy as np
import pandas as pd
import pytest

from src.dataset import (
    PRICE_COL,
    LOAD_COL,
    SOLAR_COL,
    WIND_COL,
    EXOG_COLS,
    assemble_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hourly_index():
    """A 5-day hourly DatetimeIndex in UTC, as if from blob."""
    return pd.date_range("2025-01-01", periods=5 * 24, freq="h", tz="UTC")


@pytest.fixture
def make_dam(hourly_index):
    """Function that creates a synthetic DAM DataFrame with optional truncation."""
    def _make(end_offset: int = 0) -> pd.DataFrame:
        idx = hourly_index[: len(hourly_index) - end_offset]
        return pd.DataFrame(
            {PRICE_COL: 50 + 30 * np.sin(np.arange(len(idx)) * np.pi / 12)},
            index=idx,
        )
    return _make


@pytest.fixture
def make_load(hourly_index):
    def _make(end_offset: int = 0) -> pd.DataFrame:
        idx = hourly_index[: len(hourly_index) - end_offset]
        return pd.DataFrame({LOAD_COL: 5000 + 1000 * np.cos(np.arange(len(idx)) * np.pi / 12)}, index=idx)
    return _make


@pytest.fixture
def make_renewable(hourly_index):
    def _make(end_offset: int = 0) -> pd.DataFrame:
        idx = hourly_index[: len(hourly_index) - end_offset]
        return pd.DataFrame(
            {
                SOLAR_COL: 200 + 100 * np.sin(np.arange(len(idx)) * np.pi / 12),
                WIND_COL: 1500 + 200 * np.cos(np.arange(len(idx)) * np.pi / 24),
            },
            index=idx,
        )
    return _make


# ---------------------------------------------------------------------------
# Happy path: all three aligned, inner join
# ---------------------------------------------------------------------------


class TestAssembleDatasetAligned:
    def test_returns_series_and_dataframe(self, make_dam, make_load, make_renewable):
        prices, exog = assemble_dataset(make_dam(), make_load(), make_renewable())
        assert isinstance(prices, pd.Series)
        assert isinstance(exog, pd.DataFrame)

    def test_prices_has_correct_name(self, make_dam, make_load, make_renewable):
        prices, _ = assemble_dataset(make_dam(), make_load(), make_renewable())
        assert prices.name == "price_eur_mwh"

    def test_exog_has_expected_columns(self, make_dam, make_load, make_renewable):
        _, exog = assemble_dataset(make_dam(), make_load(), make_renewable())
        assert list(exog.columns) == list(EXOG_COLS)

    def test_indexes_match(self, make_dam, make_load, make_renewable):
        prices, exog = assemble_dataset(make_dam(), make_load(), make_renewable())
        pd.testing.assert_index_equal(prices.index, exog.index)

    def test_no_nans_in_inner_join(self, make_dam, make_load, make_renewable):
        prices, exog = assemble_dataset(make_dam(), make_load(), make_renewable())
        assert prices.notna().all()
        assert exog.notna().all().all()


# ---------------------------------------------------------------------------
# Misalignment behavior
# ---------------------------------------------------------------------------


class TestAssembleDatasetMisaligned:
    """When DAM is shorter than load/renewable (the typical real case)."""

    def test_inner_join_drops_unaligned_tail(self, make_dam, make_load, make_renewable):
        # DAM ends 24h earlier
        prices, exog = assemble_dataset(
            make_dam(end_offset=24),
            make_load(),
            make_renewable(),
            join="inner",
        )
        # Both should have the shorter length
        assert len(prices) == 5 * 24 - 24
        assert prices.notna().all()
        assert exog.notna().all().all()

    def test_outer_join_keeps_all_with_nans(self, make_dam, make_load, make_renewable):
        # DAM ends 24h earlier
        prices, exog = assemble_dataset(
            make_dam(end_offset=24),
            make_load(),
            make_renewable(),
            join="outer",
        )
        # Both should have the longer length
        assert len(prices) == 5 * 24
        # The last 24 prices are NaN; exog is fully populated
        assert prices.iloc[-24:].isna().all()
        assert exog.notna().all().all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestAssembleDatasetValidation:
    def test_rejects_dam_with_wrong_column_name(self, make_load, make_renewable):
        bad_dam = pd.DataFrame(
            {"price_wrong_name": [50.0]}, index=pd.DatetimeIndex(["2025-01-01"], tz="UTC")
        )
        with pytest.raises(ValueError, match="DAM column"):
            assemble_dataset(bad_dam, make_load(), make_renewable())

    def test_rejects_load_with_two_columns(self, make_dam, make_renewable):
        bad_load = pd.DataFrame(
            {"a": [1.0], "b": [2.0]}, index=pd.DatetimeIndex(["2025-01-01"], tz="UTC")
        )
        with pytest.raises(ValueError, match="Load must have 1 column"):
            assemble_dataset(make_dam(), bad_load, make_renewable())

    def test_rejects_renewable_missing_solar(self, make_dam, make_load):
        bad_renewable = pd.DataFrame(
            {"wind_onshore": [1.0], "wrong_other": [2.0]},
            index=pd.DatetimeIndex(["2025-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="must have columns"):
            assemble_dataset(make_dam(), make_load(), bad_renewable)