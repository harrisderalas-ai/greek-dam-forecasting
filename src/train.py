"""Training pipeline for next-day DAM price forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import (
    build_supervised_dataset,
    hours_to_target,
    make_calendar_features,
    make_forecast_time_features,
    make_target_relative_lags,
    make_exogenous_target_features,
    make_exogenous_forecast_time_features,
)


@dataclass
class TrainResult:
    """A trained per-horizon model bundle plus evaluation metrics."""

    models: dict[int, lgb.LGBMRegressor]
    feature_names: list[str]
    metrics_per_horizon: pd.DataFrame
    overall_test_mae: float
    overall_test_rmse: float
    gate_closure_hour: int
    horizons: tuple[int, ...]
    same_hour_lag_days: tuple[int, ...]
    context_window: int


def temporal_split(X, y, meta, test_days: int = 14):
    cutoff = meta["forecast_time"].max() - pd.Timedelta(days=test_days)
    train_mask = meta["forecast_time"] <= cutoff
    return (
        X[train_mask].reset_index(drop=True),
        y[train_mask].reset_index(drop=True),
        meta[train_mask].reset_index(drop=True),
        X[~train_mask].reset_index(drop=True),
        y[~train_mask].reset_index(drop=True),
        meta[~train_mask].reset_index(drop=True),
    )


def train_per_horizon_models(
    prices: pd.Series,
    exog: pd.DataFrame | None = None,
    gate_closure_hour: int = 12,
    horizons: tuple[int, ...] = tuple(range(0, 24)),
    same_hour_lag_days: tuple[int, ...] = (1, 2, 7),
    context_window: int = 1,
    test_days: int = 14,
    lgbm_params: dict | None = None,
) -> TrainResult:
    if lgbm_params is None:
        lgbm_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1,
        }

    X, y, meta = build_supervised_dataset(
        prices,
        exog=exog,
        gate_closure_hour=gate_closure_hour,
        horizons=horizons,
        same_hour_lag_days=same_hour_lag_days,
        context_window=context_window,
    )
    X_tr, y_tr, m_tr, X_te, y_te, m_te = temporal_split(X, y, meta, test_days)

    models: dict[int, lgb.LGBMRegressor] = {}
    metrics_rows = []
    feature_names: list[str] | None = None

    for h in horizons:
        h_train_mask = (m_tr["horizon"] == h)
        h_test_mask = (m_te["horizon"] == h)
        if not h_train_mask.any() or not h_test_mask.any():
            continue

        X_tr_h = X_tr[h_train_mask].drop(columns=["horizon"])
        y_tr_h = y_tr[h_train_mask]
        X_te_h = X_te[h_test_mask].drop(columns=["horizon"])
        y_te_h = y_te[h_test_mask]

        if feature_names is None:
            feature_names = list(X_tr_h.columns)

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_tr_h, y_tr_h)
        models[h] = model

        pred = model.predict(X_te_h)
        metrics_rows.append({
            "horizon": h,
            "mae": mean_absolute_error(y_te_h, pred),
            "rmse": np.sqrt(mean_squared_error(y_te_h, pred)),
            "n_test": len(y_te_h),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    weights = metrics_df["n_test"].to_numpy()

    return TrainResult(
        models=models,
        feature_names=feature_names or [],
        metrics_per_horizon=metrics_df,
        overall_test_mae=float(np.average(metrics_df["mae"], weights=weights)),
        overall_test_rmse=float(np.average(metrics_df["rmse"], weights=weights)),
        gate_closure_hour=gate_closure_hour,
        horizons=horizons,
        same_hour_lag_days=same_hour_lag_days,
        context_window=context_window,
    )


def predict_next_day(
    result: TrainResult,
    prices: pd.Series,
    exog: pd.DataFrame | None = None,
    forecast_time: pd.Timestamp | None = None,
) -> pd.Series:
    """Produce a 24-hour forecast at `forecast_time`."""
    if forecast_time is None:
        candidates = prices.index[prices.index.hour == result.gate_closure_hour]
        if len(candidates) == 0:
            raise ValueError(
                f"No timestamps at gate_closure_hour={result.gate_closure_hour}."
            )
        forecast_time = candidates.max()

    forecast_times = pd.DatetimeIndex([forecast_time])
    ft = make_forecast_time_features(prices, forecast_times)

    if exog is not None:
        ft_exog = make_exogenous_forecast_time_features(exog, forecast_times)
        ft = pd.concat([ft, ft_exog], axis=1)

    preds = {}
    for h, model in result.models.items():
        distance = hours_to_target(result.gate_closure_hour, h)
        target_time = forecast_time + pd.Timedelta(hours=distance)

        tr = make_target_relative_lags(
            prices,
            forecast_times,
            horizon=h,
            gate_closure_hour=result.gate_closure_hour,
            same_hour_lag_days=result.same_hour_lag_days,
            context_window=result.context_window,
        )
        cal = make_calendar_features(pd.DatetimeIndex([target_time]))
        cal.index = ft.index

        if exog is not None:
            ex = make_exogenous_target_features(
                exog, forecast_times, horizon=h, gate_closure_hour=result.gate_closure_hour
            )
            row = pd.concat([ft, tr, cal, ex], axis=1)
        else:
            row = pd.concat([ft, tr, cal], axis=1)

        row = row[result.feature_names]
        preds[target_time] = float(model.predict(row)[0])

    return pd.Series(preds, name="prediction_eur_mwh").sort_index()