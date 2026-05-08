"""Fetch Greek day-ahead market (DAM) prices from ENTSO-E."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def load_entsoe_token() -> str:
    """
    Load the ENTSO-E API token from the appropriate source.

    Priority order:
    1. Azure Key Vault — if KEY_VAULT_NAME env var is set
       (the production path, used in Azure ML compute)
    2. Local .env file — if KEY_VAULT_NAME is not set
       (the development path, used on your laptop)

    The DefaultAzureCredential handles auth: locally via `az login`,
    in Azure compute via the managed identity.
    """
    import os

    key_vault_name = os.getenv("KEY_VAULT_NAME")

    if key_vault_name:
        # Production path: fetch from Key Vault
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        print(f"Fetching ENTSO-E API token from Azure Key Vault: {key_vault_name}")
        vault_url = f"https://{key_vault_name}.vault.azure.net"
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        secret = client.get_secret("ENTSOE-API-TOKEN")
        return secret.value

    # Development path: fetch from .env
    from dotenv import load_dotenv

    load_dotenv()
    token = os.getenv("ENTSOE_API_TOKEN")
    if not token:
        raise ValueError(
            "ENTSO-E token not found. Set ENTSOE_API_TOKEN in .env "
            "or set KEY_VAULT_NAME to read from Azure Key Vault."
        )
    return token

def fetch_dam_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = "GR",
    api_token: str | None = None,
) -> pd.Series:
    """
    Fetch day-ahead market prices for a country.

    Returns hourly prices. ENTSO-E may return 15-minute granularity for some
    periods (post Oct 2025); these are aggregated to hourly via mean.
    """
    from entsoe import EntsoePandasClient

    if api_token is None:
        api_token = load_entsoe_token()

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
        api_token = load_entsoe_token()

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
        api_token = load_entsoe_token()

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


def upload_local_to_blob(
    local_path: str | Path,
    storage_account: str,
    container: str,
    blob_name: str,
) -> str:
    """
    Upload a local file to Azure Blob Storage.

    Returns the blob URL on success.

    Authentication uses DefaultAzureCredential — your `az login` session
    locally, or the managed identity when running in Azure compute.
    """
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)

    return blob_client.url


def list_blobs_in_prefix(
    storage_account: str,
    container: str,
    prefix: str,
) -> list[str]:
    """
    List all blob names in `container` whose name starts with `prefix`.

    Returns blob names sorted alphabetically (which is also chronologically
    if names follow our YYYY-MM-DD convention).
    """
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service.get_container_client(container)

    blob_names = [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
    return sorted(blob_names)


def read_csv_from_blob(
    storage_account: str,
    container: str,
    blob_name: str,
    parse_index_as_datetime: bool = True,
) -> pd.DataFrame:
    """
    Download a CSV blob and return it as a pandas DataFrame.

    The first column of the CSV is used as the index. If
    `parse_index_as_datetime=True`, the index is parsed as UTC datetime.
    """
    import io

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    raw_bytes = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(raw_bytes), index_col=0)

    if parse_index_as_datetime:
        df.index = pd.to_datetime(df.index, utc=True)

    return df


def upload_dataframe_to_blob(
    df: pd.DataFrame,
    storage_account: str,
    container: str,
    blob_name: str,
) -> str:
    """
    Upload a DataFrame as CSV directly to a blob.

    No local file is written. The DataFrame's index is included
    (matching the format expected by `read_csv_from_blob`).

    Returns the blob URL.
    """
    import io

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    credential = DefaultAzureCredential()
    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)

    return blob_client.url
