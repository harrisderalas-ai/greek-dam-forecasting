"""Tests for src/train.py."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import build_supervised_dataset
from src.train import (
    TrainResult,
    predict_next_day,
    temporal_split,
)

# ---------------------------------------------------------------------------
# temporal_split — a small but critical helper
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    """Tests for the chronological train/test splitter."""

    def test_test_set_size_matches_test_days(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        _, _, _, _, _, m_te = temporal_split(X, y, meta, test_days=14)

        # Test set spans exactly 14 forecast days
        assert m_te["forecast_time"].nunique() == 14

        # Each forecast day produces 23-24 horizons (DST days drop the missing hour)
        assert len(m_te) <= 14 * 24
        assert len(m_te) >= 14 * 24 - 2, (
            f"Test set lost too many rows: {len(m_te)} (expected close to {14 * 24})"
        )

    def test_train_and_test_are_disjoint_in_time(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        _, _, m_tr, _, _, m_te = temporal_split(X, y, meta, test_days=14)

        # Every train forecast_time must be strictly before every test forecast_time
        latest_train = m_tr["forecast_time"].max()
        earliest_test = m_te["forecast_time"].min()
        assert latest_train < earliest_test, (
            "Train and test sets overlap in time — this is data leakage!"
        )

    def test_split_lengths_sum_to_total(self, sample_prices):
        X, y, meta = build_supervised_dataset(sample_prices)
        X_tr, _, _, X_te, _, _ = temporal_split(X, y, meta, test_days=14)
        assert len(X_tr) + len(X_te) == len(X)


# ---------------------------------------------------------------------------
# train_per_horizon_models — schema/contract tests
# ---------------------------------------------------------------------------


class TestTrainPerHorizonModelsContract:
    """Tests that the trained bundle has the expected structure."""

    def test_returns_trainresult(self, trained_result):
        assert isinstance(trained_result, TrainResult)

    def test_has_24_models(self, trained_result):
        assert len(trained_result.models) == 24

    def test_models_keyed_by_horizon(self, trained_result):
        assert set(trained_result.models.keys()) == set(range(24))

    def test_each_model_is_lgbm(self, trained_result):
        for h, model in trained_result.models.items():
            assert isinstance(model, lgb.LGBMRegressor), (
                f"Model at horizon {h} is not an LGBMRegressor"
            )

    def test_metrics_dataframe_has_expected_columns(self, trained_result):
        df = trained_result.metrics_per_horizon
        expected = {"horizon", "mae", "rmse", "n_test"}
        assert expected.issubset(df.columns)

    def test_metrics_one_row_per_horizon(self, trained_result):
        df = trained_result.metrics_per_horizon
        assert len(df) == 24
        assert set(df["horizon"]) == set(range(24))

    def test_overall_metrics_are_finite(self, trained_result):
        assert np.isfinite(trained_result.overall_test_mae)
        assert np.isfinite(trained_result.overall_test_rmse)
        assert trained_result.overall_test_mae > 0
        assert trained_result.overall_test_rmse > 0

    def test_feature_names_match_model_inputs(self, trained_result):
        # The feature_names should match what each model expects
        first_model = trained_result.models[0]
        # LGBM stores feature names from training
        expected_names = trained_result.feature_names
        actual_names = list(first_model.feature_name_)
        assert actual_names == expected_names

    def test_config_preserved_in_result(self, trained_result):
        # The TrainResult should remember how it was trained
        assert trained_result.gate_closure_hour == 12
        assert trained_result.horizons == tuple(range(0, 24))
        assert trained_result.same_hour_lag_days == (1, 2, 7)
        assert trained_result.context_window == 1


# ---------------------------------------------------------------------------
# predict_next_day — output contract
# ---------------------------------------------------------------------------


class TestPredictNextDayContract:
    """Tests that predict_next_day produces a well-formed forecast."""

    def test_returns_series(self, trained_result, sample_prices_session):
        forecast = predict_next_day(trained_result, sample_prices_session)
        assert isinstance(forecast, pd.Series)

    def test_returns_24_values(self, trained_result, sample_prices_session):
        forecast = predict_next_day(trained_result, sample_prices_session)
        assert len(forecast) == 24

    def test_forecast_index_is_tz_aware(self, trained_result, sample_prices_session):
        forecast = predict_next_day(trained_result, sample_prices_session)
        assert forecast.index.tz is not None

    def test_forecast_covers_24_consecutive_hours(self, trained_result, sample_prices_session):
        forecast = predict_next_day(trained_result, sample_prices_session)
        # Each consecutive pair should be exactly 1 hour apart
        deltas = forecast.index.to_series().diff().dropna()
        assert (deltas == pd.Timedelta(hours=1)).all()

    def test_forecast_values_are_finite(self, trained_result, sample_prices_session):
        forecast = predict_next_day(trained_result, sample_prices_session)
        assert forecast.notna().all()
        assert np.isfinite(forecast.values).all()

    def test_forecast_values_in_plausible_range(self, trained_result, sample_prices_session):
        """Predictions should be in roughly the same range as training prices."""
        forecast = predict_next_day(trained_result, sample_prices_session)
        train_min, train_max = sample_prices_session.min(), sample_prices_session.max()
        # Allow some extrapolation, but not wildly absurd
        margin = (train_max - train_min) * 0.5
        assert forecast.min() > train_min - margin
        assert forecast.max() < train_max + margin


# ---------------------------------------------------------------------------
# Quality guardrail — model must beat naïve baseline by some margin
# ---------------------------------------------------------------------------


class TestModelBeatsNaiveBaseline:
    """
    Critical ML test: the trained model must outperform a simple naïve
    baseline by at least some margin on synthetic data. If this fails,
    something is fundamentally broken in the training pipeline.
    """

    # Threshold: model must beat naïve average by at least this fraction
    # On our deterministic mock data with daily+weekly seasonality + noise,
    # a working LGBM should beat naïve average by 10-30%. We require >5%.
    MIN_RELATIVE_IMPROVEMENT = 0.05

    def test_model_beats_naive_average(self, trained_result, sample_prices_session):
        from src.baselines import evaluate_naive_baselines

        # Recompute the same train/test split that training used
        X, y, meta = build_supervised_dataset(
            sample_prices_session,
            gate_closure_hour=trained_result.gate_closure_hour,
            horizons=trained_result.horizons,
            same_hour_lag_days=trained_result.same_hour_lag_days,
            context_window=trained_result.context_window,
        )
        _, _, _, _, y_te, m_te = temporal_split(X, y, meta, test_days=14)

        naive = evaluate_naive_baselines(
            sample_prices_session,
            m_te,
            y_te,
            gate_closure_hour=trained_result.gate_closure_hour,
        )
        weights = naive["n_test"]
        naive_avg_mae = (naive["mae_average"] * weights).sum() / weights.sum()

        improvement = (naive_avg_mae - trained_result.overall_test_mae) / naive_avg_mae

        assert improvement >= self.MIN_RELATIVE_IMPROVEMENT, (
            f"Model is barely beating naïve baseline. "
            f"Model MAE: {trained_result.overall_test_mae:.2f}, "
            f"Naïve average MAE: {naive_avg_mae:.2f}, "
            f"Improvement: {improvement * 100:.1f}% "
            f"(required: >={self.MIN_RELATIVE_IMPROVEMENT * 100:.0f}%)"
        )
