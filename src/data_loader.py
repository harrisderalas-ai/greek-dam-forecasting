"""Fetch Greek day-ahead market (DAM) prices from ENTSO-E."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def fetch_dam_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = "GR",
    api_token: str | None = None,
) -> pd.Series:
    """
    Fetch day-ahead market prices for a given country and time range.

    Parameters
    ----------
    start : pd.Timestamp
        Start of the period (timezone-aware).
    end : pd.Timestamp
        End of the period (timezone-aware, exclusive).
    country_code : str
        ENTSO-E country code. Default 'GR' for Greece.
    api_token : str, optional
        ENTSO-E API token. If None, read from ENTSOE_API_TOKEN env variable.

    Returns
    -------
    pd.Series
        Hourly DAM prices in EUR/MWh, indexed by timezone-aware timestamps.
    """
    from entsoe import EntsoePandasClient

    if api_token is None:
        load_dotenv()
        api_token = os.getenv("ENTSOE_API_TOKEN")

    if not api_token:
        raise ValueError(
            "ENTSO-E API token not found. Set ENTSOE_API_TOKEN in your .env file."
        )

    client = EntsoePandasClient(api_key=api_token)
    prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    prices.name = "price_eur_mwh"
    return prices


def generate_mock_dam_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 42,
) -> pd.Series:
    """
    Generate synthetic hourly DAM prices for development/testing.

    Mimics realistic patterns: daily peaks (morning + evening),
    weekend dips, weekly seasonality, and noise.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq="h", tz=start.tz, inclusive="left")

    base = 100.0  # base price EUR/MWh
    hour_of_day = idx.hour.to_numpy()
    day_of_week = idx.dayofweek.to_numpy()

    # Daily pattern: low at night, peaks at 8am and 7pm
    daily = 30 * np.sin((hour_of_day - 7) * np.pi / 12) + 20 * np.sin(
        (hour_of_day - 18) * np.pi / 6
    )
    # Weekly pattern: lower on weekends
    weekly = np.where(day_of_week >= 5, -15, 0)
    # Random noise
    noise = rng.normal(0, 8, len(idx))

    prices = base + daily + weekly + noise
    return pd.Series(prices, index=idx, name="price_eur_mwh")


def save_prices(prices: pd.Series, path: str | Path) -> None:
    """Save prices to CSV, with timezone-aware timestamp index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)


def load_prices(path: str | Path) -> pd.Series:
    """Load prices from CSV, parsing the timestamp index."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["price_eur_mwh"]
