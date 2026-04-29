"""Tests for src/features.py."""

import numpy as np
import pandas as pd
from pandas import Series
import pytest

from src.features import (
    build_supervised_dataset,
    hours_to_target,
    make_calendar_features,
    make_forecast_time_features,
    make_target_relative_lags,
)


# ---------------------------------------------------------------------------
# hours_to_target — the small but critical utility function
# ---------------------------------------------------------------------------

class TestHoursToTarget:
    """Tests for the hours_to_target helper."""

    def test_typical_case(self):
        # Forecast at noon, predicting tomorrow at noon → 24 hours
        assert hours_to_target(gate_closure_hour=12, horizon=12) == 24

    def test_first_hour_of_next_day(self):
        # gate=12, horizon=0 (tomorrow 00:00) → 12 hours
        assert hours_to_target(gate_closure_hour=12, horizon=0) == 12

    def test_last_hour_of_next_day(self):
        # gate=12, horizon=23 (tomorrow 23:00) → 35 hours
        assert hours_to_target(gate_closure_hour=12, horizon=23) == 35

    def test_different_gate_closure(self):
        # gate=17, horizon=0 → 7 hours
        assert hours_to_target(gate_closure_hour=17, horizon=0) == 7

    def test_invalid_gate_closure_hour(self):
        with pytest.raises(ValueError, match="gate_closure_hour"):
            hours_to_target(gate_closure_hour=24, horizon=0)
        with pytest.raises(ValueError, match="gate_closure_hour"):
            hours_to_target(gate_closure_hour=-1, horizon=0)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            hours_to_target(gate_closure_hour=12, horizon=24)
        with pytest.raises(ValueError, match="horizon"):
            hours_to_target(gate_closure_hour=12, horizon=-1)


# ---------------------------------------------------------------------------
# Schema tests — verify the dataset has the expected shape and columns
# ---------------------------------------------------------------------------

class TestBuildSupervisedDatasetSchema:
    """Tests verifying X, y, meta have correct shape and columns."""

    def test_returns_three_objects(self, sample_prices):
        result = build_supervised_dataset(sample_prices)
        assert len(result) == 3
        X, y, meta = result
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(meta, pd.DataFrame)

    def test_xy_lengths_match(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        assert len(X) == len(y)
        assert len(X) == len(meta)

    def test_meta_has_expected_columns(self, sample_prices):
        _, _, meta = build_supervised_dataset(sample_prices)
        assert set(meta.columns) == {"forecast_time", "horizon", "target_time"}

    def test_y_has_no_nan(self, sample_prices):
        _, y, _ = build_supervised_dataset(sample_prices)
        assert not y.isna().any(), "Target column should never contain NaN"

    def test_horizons_in_range(self, sample_prices: Series[Any]):
        _, _, meta = build_supervised_dataset(sample_prices)
        assert meta["horizon"].min() >= 0
        assert meta["horizon"].max() <= 23

    def test_default_horizons_cover_full_day(self, sample_prices):
        _, _, meta = build_supervised_dataset(sample_prices)
        # Default horizons = (0, ..., 23), all 24 should appear
        assert set(meta["horizon"].unique()) == set(range(24))

    def test_custom_horizons_respected(self, sample_prices):
        _, _, meta = build_supervised_dataset(sample_prices, horizons=(8, 18))
        assert set(meta["horizon"].unique()) == {8, 18}

    def test_feature_columns_present(self, sample_prices):
        X, _, _ = build_supervised_dataset(sample_prices)
        ft_cols = [c for c in X.columns if c.startswith("ft_")]
        tr_cols = [c for c in X.columns if c.startswith("tr_")]
        target_cols = [c for c in X.columns if c.startswith("target_")]
        assert len(ft_cols) >= 4, "Should have several ft_ features"
        assert len(tr_cols) >= 9, "Should have at least 9 tr_ features (3 days × 3 ctx)"
        assert len(target_cols) >= 5, "Should have several target_ features"
        assert "horizon" in X.columns
        
        
# ---------------------------------------------------------------------------
# Logic tests — verify the feature pipeline matches its design
# ---------------------------------------------------------------------------

class TestForecastTimeFeaturesAreConstantWithinDay:
    """ft_* features should be identical for all horizons of one forecast day."""

    def test_ft_features_constant_across_horizons(self, sample_prices):
        X, _, meta = build_supervised_dataset(sample_prices)
        # Pick a forecast day in the middle of the dataset
        chosen = meta["forecast_time"].iloc[len(meta) // 2]
        mask = meta["forecast_time"] == chosen
        day_X = X[mask]

        for col in [c for c in day_X.columns if c.startswith("ft_")]:
            assert day_X[col].nunique() == 1, (
                f"Column {col} has {day_X[col].nunique()} unique values "
                f"across horizons of one forecast day — expected 1"
            )


class TestTargetFeaturesVary:
    """target_hour should differ across horizons of one forecast day."""

    def test_target_hour_has_24_unique_values(self, sample_prices):
        X, _, meta = build_supervised_dataset(sample_prices)
        chosen = meta["forecast_time"].iloc[len(meta) // 2]
        mask = meta["forecast_time"] == chosen
        day_X = X[mask]

        # All 24 target hours should be present
        assert day_X["target_hour"].nunique() == 24


class TestTargetTimesAreCorrect:
    """target_time = forecast_time + (24 - gate_closure) + horizon hours."""

    def test_target_time_h0_is_next_day_midnight(self, sample_prices):
        gc = 12
        _, _, meta = build_supervised_dataset(sample_prices, gate_closure_hour=gc)
        h0 = meta[meta["horizon"] == 0]
        # For each forecast time, the target should be the same day's date + 1, at 00:00
        for _, row in h0.iterrows():
            ft = row["forecast_time"]
            tt = row["target_time"]
            expected = ft.normalize() + pd.Timedelta(days=1)  # next midnight
            assert tt == expected, f"Expected {expected}, got {tt}"

    def test_target_time_h23_is_next_day_23h(self, sample_prices):
        gc = 12
        _, _, meta = build_supervised_dataset(sample_prices, gate_closure_hour=gc)
        h23 = meta[meta["horizon"] == 23]
        for _, row in h23.iterrows():
            ft = row["forecast_time"]
            tt = row["target_time"]
            expected = ft.normalize() + pd.Timedelta(days=1, hours=23)
            assert tt == expected


# ---------------------------------------------------------------------------
# THE leakage test — the most important test in the suite
# ---------------------------------------------------------------------------

class TestNoLeakage:
    """Critical: no feature value at test time may use information from after t."""

    def test_no_tr_feature_uses_post_t_data(self, sample_prices):
        """
        Programmatically verify the leakage rule for all tr_* features.

        For every (forecast_time, horizon) row, every non-NaN tr_* feature
        must correspond to a price at a timestamp <= forecast_time.
        """
        gc = 12
        prices = sample_prices
        X, _, meta = build_supervised_dataset(prices, gate_closure_hour=gc)

        # We can't directly check "what timestamp was used" — but we CAN
        # reconstruct it from the column name and the (forecast_time, horizon).
        # Column format: tr_lag_d{D}_k{K}
        tr_cols = [c for c in X.columns if c.startswith("tr_lag_")]

        for col in tr_cols:
            # Parse d and k from the column name
            # Format: tr_lag_d{D}_k{K} where K can be -1, +0, +1, etc.
            after_d = col[len("tr_lag_d"):]
            d_str, k_str = after_d.split("_k")
            d = int(d_str)
            k = int(k_str)

            offset_hours = 24 * d - k

            # For each row, compute feature_time and verify <= forecast_time
            for idx in [0, len(meta) // 2, len(meta) - 1]:
                ft = meta["forecast_time"].iloc[idx]
                h = meta["horizon"].iloc[idx]
                tt = meta["target_time"].iloc[idx]
                feature_time = tt - pd.Timedelta(hours=offset_hours)
                value = X[col].iloc[idx]

                if feature_time <= ft:
                    # Feature should have a real value (no leakage problem)
                    pass  # The fact that the value exists is fine
                else:
                    # Feature would leak — must be NaN
                    assert pd.isna(value), (
                        f"LEAKAGE: column {col} at row {idx} "
                        f"(ft={ft}, h={h}, ftime={feature_time}) "
                        f"should be NaN but has value {value}"
                    )


# ---------------------------------------------------------------------------
# NaN pattern test — codifies the table we worked out by hand
# ---------------------------------------------------------------------------

class TestNanPattern:
    """For gate_closure=12, the per-horizon NaN count in tr_* columns is fixed."""

    @pytest.mark.parametrize(
        "horizon,expected_nan_count",
        [
            (0, 0),
            (5, 0),
            (11, 0),
            (12, 1),   # d1_k+1 becomes NaN (feature_time = t + 1h)
            (13, 2),   # d1_k+0 and d1_k+1
            (14, 3),   # all three d1_*
            (20, 3),
            (23, 3),
        ],
    )
    def test_nan_count_matches_expected(
        self, sample_prices, horizon, expected_nan_count
    ):
        X, _, meta = build_supervised_dataset(
            sample_prices,
            gate_closure_hour=12,
            same_hour_lag_days=(1, 2, 7),
            context_window=1,
        )
        mask = meta["horizon"] == horizon
        rows_for_h = X[mask].reset_index(drop=True)
        # Use a row past the warm-up period
        sample = rows_for_h.iloc[10]
        tr_cols = [c for c in X.columns if c.startswith("tr_")]
        nan_count = sum(pd.isna(sample[c]) for c in tr_cols)
        assert nan_count == expected_nan_count, (
            f"At horizon={horizon}, expected {expected_nan_count} NaN tr_* cols, "
            f"got {nan_count}"
        )