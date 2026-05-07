"""Post-processing of raw ENTSO-E data into hourly-aligned form."""

from __future__ import annotations

import pandas as pd


def _detect_subhourly(index: pd.DatetimeIndex) -> bool:
    """
    Return True if the index is plausibly sub-hourly (e.g., 15-minute).

    Uses the median time delta. Robust to small gaps and mixed cadence
    (e.g., hourly for early dates, 15-min for recent dates).
    """
    if len(index) < 2:
        return False
    deltas = index.to_series().diff().dropna()
    return deltas.min() < pd.Timedelta(hours=1)


def ensure_hourly(series: pd.Series) -> pd.Series:
    """
    Normalize a Series to hourly cadence.

    Empty or single-element Series are returned unchanged. Otherwise the
    Series must have a DatetimeIndex; sub-hourly data (e.g., 15-min) is
    aggregated to hourly via mean.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"ensure_hourly expects a Series, got {type(series).__name__}")

    # Trivial inputs: pass through (cadence is irrelevant)
    if len(series) < 2:
        return series

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex")

    if not _detect_subhourly(series.index):
        return series

    return series.resample("h", label="left", closed="left").mean()


def ensure_hourly_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame to hourly cadence (each numeric column averaged).

    The DataFrame must have a DatetimeIndex. Sub-hourly data is aggregated
    to hourly via per-column mean.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"ensure_hourly_df expects a DataFrame, got {type(df).__name__}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex")

    if not _detect_subhourly(df.index):
        return df

    return df.resample("h", label="left", closed="left").mean()


def process_raw_csv(input_path: str, output_path: str) -> dict:
    """Read raw CSV, normalize, write CSV. Convenience wrapper for local files."""
    df = pd.read_csv(input_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    result, report = process_raw_dataframe(df)
    result.to_csv(output_path)
    return report
    
    
def process_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize a raw DataFrame to hourly cadence.

    Returns the processed DataFrame and a small report dict.
    Pure function — no I/O.
    """
    input_rows = len(df)
    columns = list(df.columns)

    if df.shape[1] == 1:
        result = ensure_hourly(df.iloc[:, 0]).to_frame(name=df.columns[0])
    else:
        result = ensure_hourly_df(df)

    report = {
        "input_rows": input_rows,
        "output_rows": len(result),
        "granularity_changed": input_rows != len(result),
        "columns": columns,
    }
    return result, report
