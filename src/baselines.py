"""Naïve baselines for next-day DAM price forecasting.

A baseline maps (prices, forecast_time, gate_closure_hour, horizon) to a
predicted price at target_time = forecast_time + (24 - gate_closure_hour) + horizon.

All baselines respect leakage: they only use prices at times <= forecast_time.
"""

from __future__ import annotations

import pandas as pd

from src.features import hours_to_target


def _to_utc(prices: pd.Series) -> pd.Series:
    """Return a copy of prices with a UTC index for safe lookups."""
    out = prices.copy()
    out.index = prices.index.tz_convert("UTC")
    return out


def naive_yesterday(
    prices: pd.Series,
    forecast_time: pd.Timestamp,
    gate_closure_hour: int,
    horizon: int,
) -> float:
    """
    Naïve baseline: 'yesterday at the target hour'.

    Falls back to 'two days ago at the target hour' if yesterday's value
    would leak (i.e. lie after forecast_time).
    """
    distance = hours_to_target(gate_closure_hour, horizon)
    target_time = forecast_time + pd.Timedelta(hours=distance)

    candidate_24h = target_time - pd.Timedelta(hours=24)
    if candidate_24h <= forecast_time:
        ref = candidate_24h
    else:
        ref = target_time - pd.Timedelta(hours=48)

    prices_utc = _to_utc(prices)
    return float(prices_utc.reindex([ref.tz_convert("UTC")]).iloc[0])


def naive_last_week(
    prices: pd.Series,
    forecast_time: pd.Timestamp,
    gate_closure_hour: int,
    horizon: int,
) -> float:
    """Naïve baseline: 'same hour last week' (target_time - 168h)."""
    distance = hours_to_target(gate_closure_hour, horizon)
    target_time = forecast_time + pd.Timedelta(hours=distance)
    ref = target_time - pd.Timedelta(hours=168)

    prices_utc = _to_utc(prices)
    return float(prices_utc.reindex([ref.tz_convert("UTC")]).iloc[0])


def naive_average(
    prices: pd.Series,
    forecast_time: pd.Timestamp,
    gate_closure_hour: int,
    horizon: int,
) -> float:
    """Average of yesterday-or-2d-ago and last-week naïve baselines."""
    a = naive_yesterday(prices, forecast_time, gate_closure_hour, horizon)
    b = naive_last_week(prices, forecast_time, gate_closure_hour, horizon)
    return (a + b) / 2.0


# ---------------------------------------------------------------------------
# Vectorised evaluation across the test set
# ---------------------------------------------------------------------------


def evaluate_naive_baselines(
    prices: pd.Series,
    meta_test: pd.DataFrame,
    y_test: pd.Series,
    gate_closure_hour: int,
) -> pd.DataFrame:
    """
    Evaluate all three naïve baselines against the test labels.

    Parameters
    ----------
    prices : pd.Series
        Hourly DAM prices, used to look up reference values.
    meta_test : pd.DataFrame
        Must contain columns 'forecast_time', 'horizon', 'target_time'.
    y_test : pd.Series
        Actual target prices, aligned with meta_test.

    Returns
    -------
    pd.DataFrame with columns:
        horizon, mae_yesterday, mae_last_week, mae_average,
                 rmse_yesterday, rmse_last_week, rmse_average,
                 n_test
    """
    import numpy as np

    df = meta_test.copy().reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    df["pred_yesterday"] = [
        naive_yesterday(prices, ft, gate_closure_hour, h)
        for ft, h in zip(df["forecast_time"], df["horizon"])
    ]
    df["pred_last_week"] = [
        naive_last_week(prices, ft, gate_closure_hour, h)
        for ft, h in zip(df["forecast_time"], df["horizon"])
    ]
    df["pred_average"] = (df["pred_yesterday"] + df["pred_last_week"]) / 2.0
    df["actual"] = y_test.values

    out_rows = []
    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h]
        sub = sub.dropna(subset=["pred_yesterday", "pred_last_week", "actual"])
        if len(sub) == 0:
            continue
        out_rows.append(
            {
                "horizon": h,
                "mae_yesterday": (sub["pred_yesterday"] - sub["actual"]).abs().mean(),
                "mae_last_week": (sub["pred_last_week"] - sub["actual"]).abs().mean(),
                "mae_average": (sub["pred_average"] - sub["actual"]).abs().mean(),
                "rmse_yesterday": np.sqrt(((sub["pred_yesterday"] - sub["actual"]) ** 2).mean()),
                "rmse_last_week": np.sqrt(((sub["pred_last_week"] - sub["actual"]) ** 2).mean()),
                "rmse_average": np.sqrt(((sub["pred_average"] - sub["actual"]) ** 2).mean()),
                "n_test": len(sub),
            }
        )

    return pd.DataFrame(out_rows)
