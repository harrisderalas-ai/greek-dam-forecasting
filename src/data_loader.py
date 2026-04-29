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


def fetch_load_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = "GR",
    api_token: str | None = None,
) -> pd.Series:
    """
    Fetch ENTSO-E day-ahead load forecast (MW) for a country.

    Returns hourly forecast values that were published *day-ahead*. These are
    what was known at gate-closure time, making them legitimate features.
    """
    from entsoe import EntsoePandasClient

    if api_token is None:
        load_dotenv()
        api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN not set.")

    client = EntsoePandasClient(api_key=api_token)
    df = client.query_load_forecast(country_code, start=start, end=end)
    # ENTSO-E returns a DataFrame with one or more forecast columns;
    # we want the day-ahead one
    if isinstance(df, pd.DataFrame):
        if "Forecasted Load" in df.columns:
            series = df["Forecasted Load"]
        else:
            series = df.iloc[:, 0]
    else:
        series = df
    series.name = "load_forecast_mw"
    return series


def fetch_renewable_forecast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = "GR",
    api_token: str | None = None,
) -> pd.DataFrame:
    """
    Fetch ENTSO-E day-ahead wind + solar generation forecast (MW).

    Returns a DataFrame with whatever renewable types are reported for the
    country, typically: 'Solar', 'Wind Onshore', and sometimes 'Wind Offshore'.
    """
    from entsoe import EntsoePandasClient

    if api_token is None:
        load_dotenv()
        api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise ValueError("ENTSOE_API_TOKEN not set.")

    client = EntsoePandasClient(api_key=api_token)
    df = client.query_wind_and_solar_forecast(country_code, start=start, end=end)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def fetch_all_inputs(
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = "GR",
    api_token: str | None = None,
) -> pd.DataFrame:
    """
    Fetch prices + load forecast + renewable forecasts and align them on
    a single hourly DatetimeIndex.

    Returns a DataFrame with at least: price_eur_mwh, load_forecast_mw,
    and one column per renewable type reported (e.g., solar, wind_onshore).
    """
    prices = fetch_dam_prices(start, end, country_code, api_token)
    load = fetch_load_forecast(start, end, country_code, api_token)
    renew = fetch_renewable_forecast(start, end, country_code, api_token)

    df = pd.concat(
        [prices.rename("price_eur_mwh"), load.rename("load_forecast_mw"), renew],
        axis=1,
    )
    return df


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
    """Save a price series to CSV with UTC-normalized timestamps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = prices.copy()
    if out.index.tz is None:
        raise ValueError("Refusing to save tz-naive index — wrap with tz_localize first.")
    out.index = out.index.tz_convert("UTC")
    out.to_csv(path)


def load_prices(path: str | Path, target_tz: str = "Europe/Athens") -> pd.Series:
    """Load a price series from CSV, returning a tz-aware Series."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(target_tz)
    return df["price_eur_mwh"]
