"""Feature engineering for next-day DAM price forecasting.

Convention
----------
- Forecast time t = today at gate_closure_hour (default 12:00 local).
- Horizons h in {0, ..., 23} index the target hour-of-day for tomorrow.
- For target h, target_time = t + (24 - gate_closure_hour) + h hours.
- All features may only use price information at times <= t.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

GREEK_HOLIDAYS = holidays.country_holidays("GR")


def hours_to_target(gate_closure_hour: int, horizon: int) -> int:
    """Hours between forecast time t and target time, given config."""
    if not 0 <= gate_closure_hour <= 23:
        raise ValueError(f"gate_closure_hour must be in 0..23, got {gate_closure_hour}")
    if not 0 <= horizon <= 23:
        raise ValueError(f"horizon must be in 0..23, got {horizon}")
    return (24 - gate_closure_hour) + horizon


# ---------------------------------------------------------------------------
# Calendar features (about the TARGET hour)
# ---------------------------------------------------------------------------

def make_calendar_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features for the target timestamp."""
    if target_index.tz is None:
        raise ValueError("Index must be timezone-aware.")

    df = pd.DataFrame(index=target_index)
    df["target_hour"] = target_index.hour
    df["target_dow"] = target_index.dayofweek
    df["target_month"] = target_index.month
    df["target_is_weekend"] = (target_index.dayofweek >= 5).astype(int)
    df["target_is_holiday"] = pd.Series(target_index, index=target_index).apply(
        lambda ts: ts.date() in GREEK_HOLIDAYS
    ).astype(int)

    df["target_hour_sin"] = np.sin(2 * np.pi * df["target_hour"] / 24)
    df["target_hour_cos"] = np.cos(2 * np.pi * df["target_hour"] / 24)
    df["target_dow_sin"] = np.sin(2 * np.pi * df["target_dow"] / 7)
    df["target_dow_cos"] = np.cos(2 * np.pi * df["target_dow"] / 7)
    return df


# ---------------------------------------------------------------------------
# Forecast-time features (about the MARKET STATE at t)
# ---------------------------------------------------------------------------

def make_forecast_time_features(
    prices: pd.Series,
    forecast_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Features describing the market state at forecast time t.
    Same value for all horizons predicted from the same t.

    Lag arithmetic done in UTC to avoid DST edge cases.
    """
    df = pd.DataFrame(index=forecast_times)

    prices_utc = prices.copy()
    prices_utc.index = prices.index.tz_convert("UTC")
    forecast_times_utc = forecast_times.tz_convert("UTC")

    for lag in (0, 1, 2, 3):
        ts_utc = forecast_times_utc - pd.Timedelta(hours=lag)
        df[f"ft_price_minus_{lag}h"] = prices_utc.reindex(ts_utc).values

    for w in (24, 168):
        rolled = prices.rolling(window=w, min_periods=1)
        df[f"ft_mean_{w}h"] = rolled.mean().reindex(forecast_times).values
        df[f"ft_std_{w}h"] = rolled.std().reindex(forecast_times).values
        df[f"ft_max_{w}h"] = rolled.max().reindex(forecast_times).values
        df[f"ft_min_{w}h"] = rolled.min().reindex(forecast_times).values

    return df


# ---------------------------------------------------------------------------
# Target-relative lag features (about the TARGET HOUR's recent behaviour)
# ---------------------------------------------------------------------------

def make_target_relative_lags(
    prices: pd.Series,
    forecast_times: pd.DatetimeIndex,
    horizon: int,
    gate_closure_hour: int,
    same_hour_lag_days: tuple[int, ...] = (1, 2, 7),
    context_window: int = 1,
) -> pd.DataFrame:
    """
    Target-relative lags. Always emits all configured columns.

    Disallowed lags (would leak future) are set to NaN; LightGBM handles NaN natively.
    Lag arithmetic done in UTC to avoid DST edge cases.
    """
    if context_window < 0:
        raise ValueError("context_window must be non-negative.")

    distance = hours_to_target(gate_closure_hour, horizon)

    prices_utc = prices.copy()
    prices_utc.index = prices.index.tz_convert("UTC")
    forecast_times_utc = forecast_times.tz_convert("UTC")
    target_times_utc = forecast_times_utc + pd.Timedelta(hours=distance)

    df = pd.DataFrame(index=forecast_times)

    for d in same_hour_lag_days:
        center_offset = 24 * d
        for k in range(-context_window, context_window + 1):
            offset_hours = center_offset - k
            col_name = f"tr_lag_d{d}_k{k:+d}"

            if offset_hours < distance:
                # Leakage: this hour is after forecast time t.
                df[col_name] = np.nan
            else:
                feature_times_utc = target_times_utc - pd.Timedelta(hours=offset_hours)
                df[col_name] = prices_utc.reindex(feature_times_utc).values

    return df


def make_exogenous_target_features(
    exog: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    horizon: int,
    gate_closure_hour: int,
) -> pd.DataFrame:
    """
    Forecast values for exogenous variables at the target hour.

    Since these are *forecasts* (not actuals), they're available before
    gate closure and are leakage-safe regardless of horizon.

    Parameters
    ----------
    exog : pd.DataFrame
        Exogenous data with columns like 'load_forecast_mw', 'solar', 'wind_onshore'.
        Indexed by hourly timezone-aware timestamps.
    forecast_times : pd.DatetimeIndex
        Forecast times (one per day, at gate_closure_hour).
    horizon : int
        Target hour-of-day (0..23).
    gate_closure_hour : int
        Hour at which forecast is produced.
    """
    distance = hours_to_target(gate_closure_hour, horizon)

    exog_utc = exog.copy()
    exog_utc.index = exog.index.tz_convert("UTC")
    forecast_times_utc = forecast_times.tz_convert("UTC")
    target_times_utc = forecast_times_utc + pd.Timedelta(hours=distance)

    # Look up exogenous values at the target time
    out = exog_utc.reindex(target_times_utc).reset_index(drop=True)
    out.columns = [f"target_{c}" for c in out.columns]
    out.index = forecast_times

    # Net load (load minus all renewables) — captures residual demand
    if "target_load_forecast_mw" in out.columns:
        renew_cols = [c for c in out.columns if c not in ("target_load_forecast_mw",)]
        if renew_cols:
            out["target_net_load_mw"] = (
                out["target_load_forecast_mw"] - out[renew_cols].sum(axis=1)
            )

    return out


def make_exogenous_forecast_time_features(
    exog: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Exogenous values *at the forecast time t*. Same for all horizons of one day.
    Useful for relative comparisons (e.g., 'tomorrow's solar is X% of today's').
    """
    exog_utc = exog.copy()
    exog_utc.index = exog.index.tz_convert("UTC")
    forecast_times_utc = forecast_times.tz_convert("UTC")

    out = exog_utc.reindex(forecast_times_utc).reset_index(drop=True)
    out.columns = [f"ft_{c}" for c in out.columns]
    out.index = forecast_times

    if "ft_load_forecast_mw" in out.columns:
        renew_cols = [c for c in out.columns if c not in ("ft_load_forecast_mw",)]
        if renew_cols:
            out["ft_net_load_mw"] = (
                out["ft_load_forecast_mw"] - out[renew_cols].sum(axis=1)
            )

    return out

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_supervised_dataset(
    prices: pd.Series,
    exog: pd.DataFrame | None = None,
    gate_closure_hour: int = 12,
    horizons: tuple[int, ...] = tuple(range(0, 24)),
    same_hour_lag_days: tuple[int, ...] = (1, 2, 7),
    context_window: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a supervised dataset for direct multi-horizon next-day forecasting.

    Parameters
    ----------
    prices : pd.Series
        Hourly DAM prices, timezone-aware.
    exog : pd.DataFrame, optional
        Exogenous features (load forecast, renewable forecasts) on the same
        hourly timezone-aware index as prices. If None, only price-derived
        features are used.
    gate_closure_hour : int
    horizons : tuple of int
    same_hour_lag_days : tuple of int
    context_window : int

    Returns
    -------
    X, y, meta
    """
    if prices.index.freq is None:
        prices = prices.asfreq("h")

    forecast_times = prices.index[prices.index.hour == gate_closure_hour]
    ft_features = make_forecast_time_features(prices, forecast_times)

    if exog is not None:
        ft_exog = make_exogenous_forecast_time_features(exog, forecast_times)
        ft_features = pd.concat([ft_features, ft_exog], axis=1)

    blocks, metas = [], []

    for h in horizons:
        distance = hours_to_target(gate_closure_hour, h)
        target_times = forecast_times + pd.Timedelta(hours=distance)
        valid = target_times.isin(prices.index)
        if not valid.any():
            continue

        ft_block = ft_features.loc[forecast_times[valid]].reset_index(drop=True)

        tr_lags = make_target_relative_lags(
            prices,
            forecast_times[valid],
            horizon=h,
            gate_closure_hour=gate_closure_hour,
            same_hour_lag_days=same_hour_lag_days,
            context_window=context_window,
        ).reset_index(drop=True)

        cal = make_calendar_features(target_times[valid]).reset_index(drop=True)

        if exog is not None:
            ex = make_exogenous_target_features(
                exog, forecast_times[valid], horizon=h, gate_closure_hour=gate_closure_hour
            ).reset_index(drop=True)
            block = pd.concat([ft_block, tr_lags, cal, ex], axis=1)
        else:
            block = pd.concat([ft_block, tr_lags, cal], axis=1)

        block["horizon"] = h
        block["__target__"] = prices.reindex(target_times[valid]).values

        meta = pd.DataFrame({
            "forecast_time": forecast_times[valid],
            "horizon": h,
            "target_time": target_times[valid],
        }).reset_index(drop=True)

        blocks.append(block)
        metas.append(meta)

    if not blocks:
        raise ValueError("No valid (forecast_time, horizon) pairs in the data.")

    full = pd.concat(blocks, ignore_index=True)
    meta_full = pd.concat(metas, ignore_index=True)

    must_be_present = ["__target__"] + [c for c in full.columns if c.startswith("ft_")]
    full = full.dropna(subset=must_be_present)
    meta_full = meta_full.loc[full.index].reset_index(drop=True)
    full = full.reset_index(drop=True)

    y = full["__target__"]
    X = full.drop(columns=["__target__"])

    return X, y, meta_full