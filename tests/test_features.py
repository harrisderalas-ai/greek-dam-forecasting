"""Tests for src/features.py."""

import pandas as pd
import pytest

from src.features import (
    build_supervised_dataset,
    hours_to_target,
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

    def test_horizons_in_range(self, sample_prices):
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

        # Pick a forecast_time that has all 24 horizons present
        # (some near-edge forecast_times may be missing some horizons after dropna)
        counts = meta.groupby("forecast_time").size()
        full_days = counts[counts == 24].index
        assert len(full_days) > 0, "No forecast days had all 24 horizons"
        chosen = full_days[len(full_days) // 2]  # pick middle of full days

        mask = meta["forecast_time"] == chosen
        day_X = X[mask]

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
    """Critical: no feature value at test time may use information from after
    the knowledge horizon (end of today Athens local at forecast time t)."""

    def test_no_tr_feature_uses_post_knowledge_horizon_data(self, sample_prices):
        """
        Programmatically verify the leakage rule for all tr_* features.

        For every (forecast_time, horizon) row, every non-NaN tr_* feature
        must correspond to a price at a timestamp strictly before
        end-of-today Athens local (the knowledge horizon at gate closure).
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
            after_d = col[len("tr_lag_d") :]
            d_str, k_str = after_d.split("_k")
            d = int(d_str)
            k = int(k_str)

            offset_hours = 24 * d - k

            # For each row, compute feature_time and verify it's strictly
            # before end-of-today Athens (the knowledge horizon).
            for idx in [0, len(meta) // 2, len(meta) - 1]:
                ft = meta["forecast_time"].iloc[idx]
                h = meta["horizon"].iloc[idx]
                tt = meta["target_time"].iloc[idx]
                feature_time = tt - pd.Timedelta(hours=offset_hours)

                # Knowledge horizon: end-of-today Athens (in same tz as ft)
                ft_athens = ft.tz_convert("Europe/Athens")
                knowledge_horizon = ft_athens.normalize() + pd.Timedelta(days=1)
                # Compare in a consistent timezone
                feature_time_athens = feature_time.tz_convert("Europe/Athens")

                value = X[col].iloc[idx]

                if feature_time_athens < knowledge_horizon:
                    # Feature should have a real value (no leakage)
                    pass  # value being present is fine
                else:
                    # Feature would leak — must be NaN
                    assert pd.isna(value), (
                        f"LEAKAGE: column {col} at row {idx} "
                        f"(ft={ft}, h={h}, ftime={feature_time}, "
                        f"knowledge_horizon={knowledge_horizon}) "
                        f"should be NaN but has value {value}"
                    )


# ---------------------------------------------------------------------------
# NaN pattern test — codifies the table we worked out by hand
# ---------------------------------------------------------------------------


class TestNanPattern:
    """Under the new knowledge model (today's prices are fully known at
    gate closure), tr_* features have very few NaNs. Only the column that
    looks at "1 hour past target's same-hour-yesterday" can leak, and only
    for horizon=23 (which would point at tomorrow 00:00).
    """

    @pytest.mark.parametrize(
        "horizon,expected_nan_count",
        [
            (0, 0),
            (5, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (20, 0),
            (22, 0),
            (23, 1),  # tr_lag_d1_k+1 points at tomorrow 00:00 → NaN
        ],
    )
    def test_nan_count_matches_expected(self, sample_prices, horizon, expected_nan_count):
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
            f"At horizon={horizon}, expected {expected_nan_count} NaN tr_* cols, got {nan_count}"
        )


class TestTodayPricesAvailableAtGateClosure:
    """At gate closure today, today's full 24 prices are already published
    (yesterday's DAM auction released them). Features that point at today's
    hours should NOT be NaN, even if those hours are after gate_closure_hour."""

    def test_d1_lag_for_evening_horizons_is_not_nan(self, sample_prices):
        """
        For predicting tomorrow's evening (e.g., horizon=20), the d=1 same-hour
        lag points at TODAY's 20:00. That price was published yesterday and is
        known at gate closure (12:00 today). It must not be NaN.
        """
        X, _, meta = build_supervised_dataset(
            sample_prices,
            gate_closure_hour=12,
            same_hour_lag_days=(1,),
            context_window=0,
        )

        # Pick a horizon=20 row past the warm-up period
        mask = meta["horizon"] == 20
        rows = X[mask].reset_index(drop=True)
        assert len(rows) > 10, "Need enough data past warm-up"

        sample = rows.iloc[10]
        assert not pd.isna(sample["tr_lag_d1_k+0"]), (
            "tr_lag_d1_k+0 at horizon=20 should be today's 20:00 price, "
            "which is published at gate closure and must not be NaN"
        )

    def test_only_d1_k_plus_1_at_h23_is_nan(self, sample_prices):
        """
        The ONLY tr_* feature that can be NaN under the new model is
        tr_lag_d1_k+1 at horizon=23, because it would point at tomorrow 00:00
        (which is past the knowledge horizon).
        """
        X, _, meta = build_supervised_dataset(
            sample_prices,
            gate_closure_hour=12,
            same_hour_lag_days=(1, 2, 7),
            context_window=1,
        )
        tr_cols = [c for c in X.columns if c.startswith("tr_")]

        # For horizon=23, exactly one tr_ feature should be NaN: tr_lag_d1_k+1
        mask = meta["horizon"] == 23
        rows = X[mask].reset_index(drop=True)
        sample = rows.iloc[10]
        nan_cols = [c for c in tr_cols if pd.isna(sample[c])]
        assert nan_cols == ["tr_lag_d1_k+1"], (
            f"At horizon=23, expected only tr_lag_d1_k+1 to be NaN, "
            f"got {nan_cols}"
        )

        # For horizon=10, no tr_ feature should be NaN
        mask = meta["horizon"] == 10
        rows = X[mask].reset_index(drop=True)
        sample = rows.iloc[10]
        nan_cols = [c for c in tr_cols if pd.isna(sample[c])]
        assert nan_cols == [], (
            f"At horizon=10, expected no NaN tr_* features, got {nan_cols}"
        )
        
        
class TestDSTHandling:
    """Verify the pipeline handles DST transitions correctly.

    On spring-forward day (March 31): the local-clock 03:00 hour does not exist.
    On fall-back day (October 27): the local-clock 03:00 hour exists twice.

    The pipeline indexes horizons by UTC-distance from forecast_time, so it
    always produces N horizons regardless of local-clock anomalies. The
    assertions here reflect that.
    """

    def test_normal_day_has_24_unique_local_hours(self, dst_spanning_prices):
        """Sanity check on a non-DST day."""
        from src.features import build_supervised_dataset

        _, _, meta = build_supervised_dataset(dst_spanning_prices, gate_closure_hour=12)
        forecast_time = pd.Timestamp("2024-03-20 12:00", tz="Europe/Athens")
        rows = meta[meta["forecast_time"] == forecast_time]

        assert len(rows) == 24
        local_hours = sorted({int(t.hour) for t in rows["target_time"]})
        assert local_hours == list(range(24))

    def test_spring_forward_day_skips_local_3am(self, dst_spanning_prices):
        """
        On the day before spring-forward, local-clock 03:00 doesn't exist on
        the target day. The pipeline should still produce 24 forecasts (UTC
        spacing), but the unique local-hours should be only 23 (3 is missing).
        """
        from src.features import build_supervised_dataset

        _, _, meta = build_supervised_dataset(dst_spanning_prices, gate_closure_hour=12)
        forecast_time = pd.Timestamp("2024-03-30 12:00", tz="Europe/Athens")
        rows = meta[meta["forecast_time"] == forecast_time]

        # The pipeline produces 24 horizons (UTC-spaced)
        assert len(rows) == 24

        # But local-clock hours cover only 23 unique values (3am skipped)
        local_hours = sorted({int(t.hour) for t in rows["target_time"]})
        assert len(local_hours) == 23
        assert 3 not in local_hours

    def test_spring_forward_day_pipeline_does_not_crash(self, dst_spanning_prices):
        """The full pipeline should run on data covering spring-forward."""
        from src.features import build_supervised_dataset
        from src.train import train_per_horizon_models

        X, y, meta = build_supervised_dataset(dst_spanning_prices, gate_closure_hour=12)
        assert len(X) > 0
        assert not y.isna().any()

        fast_params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "random_state": 42,
            "verbose": -1,
        }
        result = train_per_horizon_models(
            dst_spanning_prices,
            gate_closure_hour=12,
            horizons=tuple(range(0, 24)),
            test_days=7,
            lgbm_params=fast_params,
        )
        assert len(result.models) == 24

    def test_fall_back_day_pipeline_does_not_crash(self, fall_back_prices):
        """The full pipeline should run on data covering fall-back."""
        from src.features import build_supervised_dataset
        from src.train import train_per_horizon_models

        X, y, meta = build_supervised_dataset(fall_back_prices, gate_closure_hour=12)
        assert len(X) > 0
        assert not y.isna().any()

        fast_params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "random_state": 42,
            "verbose": -1,
        }
        result = train_per_horizon_models(
            fall_back_prices,
            gate_closure_hour=12,
            horizons=tuple(range(0, 24)),
            test_days=7,
            lgbm_params=fast_params,
        )
        assert len(result.models) == 24

    def test_fall_back_day_produces_24_forecasts(self, fall_back_prices):
        """
        On the day before fall-back, the pipeline produces 24 forecasts.
        Local-clock 03:00 appears TWICE on the target day (with different UTC
        offsets), so unique local hours are only 23 — but we still produce 24
        distinct UTC-spaced forecasts.
        """
        from src.features import build_supervised_dataset

        _, _, meta = build_supervised_dataset(fall_back_prices, gate_closure_hour=12)
        forecast_time = pd.Timestamp("2024-10-26 12:00", tz="Europe/Athens")
        rows = meta[meta["forecast_time"] == forecast_time]

        # 24 horizons regardless of DST
        assert len(rows) == 24

        # Target times themselves (UTC-distinct) should all differ
        assert rows["target_time"].nunique() == 24

        # But unique local-clock hours = 23 (3am appears twice)
        local_hours = sorted({int(t.hour) for t in rows["target_time"]})
        assert len(local_hours) == 23

        # Hour 3 should be duplicated — both offsets in the target_time list
        hour_3_count = sum(1 for t in rows["target_time"] if t.hour == 3)
        assert hour_3_count == 2
