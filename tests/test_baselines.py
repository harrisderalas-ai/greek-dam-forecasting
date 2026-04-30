"""Tests for src/baselines.py."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    naive_yesterday,
    naive_last_week,
    naive_average,
    evaluate_naive_baselines,
)
from src.features import build_supervised_dataset
from src.train import temporal_split


# ---------------------------------------------------------------------------
# Single-prediction tests
# ---------------------------------------------------------------------------

class TestNaiveYesterday:
    """Tests for the 'yesterday at target hour' baseline."""

    def test_returns_a_finite_number(self, sample_prices):
        forecast_time = pd.Timestamp("2024-02-15 12:00", tz="Europe/Athens")
        pred = naive_yesterday(
            sample_prices, forecast_time, gate_closure_hour=12, horizon=8
        )
        assert isinstance(pred, float)
        assert np.isfinite(pred)

    def test_returns_correct_value_for_short_horizon(self, sample_prices):
        # For h=0 with gc=12: target = next day 00:00
        # yesterday-same-hour = today 00:00 (24h before next-day midnight)
        forecast_time = pd.Timestamp("2024-02-15 12:00", tz="Europe/Athens")
        target_time = pd.Timestamp("2024-02-16 00:00", tz="Europe/Athens")
        expected_ref = target_time - pd.Timedelta(hours=24)  # = 2024-02-15 00:00

        pred = naive_yesterday(
            sample_prices, forecast_time, gate_closure_hour=12, horizon=0
        )
        assert pred == sample_prices.loc[expected_ref]

    def test_falls_back_to_two_days_ago_on_leakage(self, sample_prices):
        # For h=18 with gc=12: target = next day 18:00
        # yesterday-same-hour = today 18:00 (LEAKAGE — after t=12:00)
        # So function should use 2-days-ago = yesterday 18:00
        forecast_time = pd.Timestamp("2024-02-15 12:00", tz="Europe/Athens")
        target_time = pd.Timestamp("2024-02-16 18:00", tz="Europe/Athens")
        expected_ref = target_time - pd.Timedelta(hours=48)  # = 2024-02-14 18:00

        pred = naive_yesterday(
            sample_prices, forecast_time, gate_closure_hour=12, horizon=18
        )
        assert pred == sample_prices.loc[expected_ref]


class TestNaiveLastWeek:
    """Tests for the 'same hour last week' baseline."""

    def test_returns_value_at_target_minus_168h(self, sample_prices):
        forecast_time = pd.Timestamp("2024-02-15 12:00", tz="Europe/Athens")
        target_time = pd.Timestamp("2024-02-16 18:00", tz="Europe/Athens")
        expected_ref = target_time - pd.Timedelta(hours=168)

        pred = naive_last_week(
            sample_prices, forecast_time, gate_closure_hour=12, horizon=18
        )
        assert pred == sample_prices.loc[expected_ref]


class TestNaiveAverage:
    """The average baseline should equal the mean of the two underlying baselines."""

    def test_equals_mean_of_components(self, sample_prices):
        forecast_time = pd.Timestamp("2024-02-15 12:00", tz="Europe/Athens")
        h = 10

        ny = naive_yesterday(sample_prices, forecast_time, 12, h)
        nlw = naive_last_week(sample_prices, forecast_time, 12, h)
        navg = naive_average(sample_prices, forecast_time, 12, h)

        assert navg == pytest.approx((ny + nlw) / 2)


# ---------------------------------------------------------------------------
# Vectorized evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluateNaiveBaselines:
    """Tests for the bulk evaluator that runs over a test set."""

    def test_returns_dataframe_with_expected_columns(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        _, _, _, _, y_te, m_te = temporal_split(X, y, meta, test_days=14)

        result = evaluate_naive_baselines(
            sample_prices, m_te, y_te, gate_closure_hour=12
        )

        expected_cols = {
            "horizon", "mae_yesterday", "mae_last_week", "mae_average",
            "rmse_yesterday", "rmse_last_week", "rmse_average", "n_test",
        }
        assert expected_cols.issubset(result.columns)

    def test_one_row_per_horizon(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        _, _, _, _, y_te, m_te = temporal_split(X, y, meta, test_days=14)

        result = evaluate_naive_baselines(
            sample_prices, m_te, y_te, gate_closure_hour=12
        )
        assert len(result) == 24
        assert set(result["horizon"]) == set(range(24))

    def test_all_metrics_are_positive(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        _, _, _, _, y_te, m_te = temporal_split(X, y, meta, test_days=14)

        result = evaluate_naive_baselines(
            sample_prices, m_te, y_te, gate_closure_hour=12
        )
        for col in ["mae_yesterday", "mae_last_week", "mae_average"]:
            assert (result[col] > 0).all(), f"{col} should be strictly positive"
            assert result[col].notna().all()