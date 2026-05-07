"""
One-time backfill: fetch historical ENTSO-E data and upload to Blob Storage.

Usage:
    python -m scripts.backfill_historical

Outputs three blobs in the `raw` container, each spanning from 2023-01-01
through end-of-tomorrow:
    dam_prices/historical_2023-01-01_to_<DAM_END>.csv
    load_forecast/historical_2023-01-01_to_<TOMORROW>.csv
    renewable_forecast/historical_2023-01-01_to_<TOMORROW>.csv

Date conventions:
    - Load and renewable forecasts publish day-ahead. Tomorrow's forecast is
      always available, sometimes the day-after-tomorrow's too. We fetch
      through end-of-tomorrow.
    - DAM publishes day-ahead at noon. Before noon today, only today's prices
      are public. After noon today, tomorrow's prices are public too. The
      script fetches through end-of-tomorrow regardless; whatever's published
      gets returned.
    - DAM_END in the blob name reflects the actual final date returned, which
      may be today or tomorrow depending on when the script runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_loader import (
    fetch_dam_prices,
    fetch_load_forecast,
    fetch_renewable_forecast,
    upload_local_to_blob,
)

# Configuration
STORAGE_ACCOUNT = "sagreekdamdevweu"
CONTAINER = "raw"
COUNTRY_CODE = "GR"
TIMEZONE = "Europe/Athens"
HISTORY_START = pd.Timestamp("2023-01-01", tz=TIMEZONE)
LOCAL_TMP_DIR = Path("data/raw/backfill_tmp")


def _historical_blob_name(data_type: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Build the blob path: '<data_type>/historical_<start>_to_<end>.csv'."""
    s = start.strftime("%Y-%m-%d")
    e = end.strftime("%Y-%m-%d")
    return f"{data_type}/historical_{s}_to_{e}.csv"


def backfill_dam_prices(today: pd.Timestamp) -> None:
    """
    Fetch DAM prices from HISTORY_START through end-of-yesterday.

    We deliberately exclude today and tomorrow's prices, even if they're
    publicly available, because the inference time is BEFORE today's
    gate closure (at 13:00 Athens). This keeps the historical archive
    consistent with what the model would observe at inference time.
    """
    end = today  # exclusive — fetches up to (but not including) today 00:00 Athens
    end_label = today - pd.Timedelta(days=1)  # last full day in the data

    print(f"[DAM] Fetching {HISTORY_START.date()} → {end_label.date()} (yesterday)")
    series = fetch_dam_prices(start=HISTORY_START, end=end, country_code=COUNTRY_CODE)
    print(f"[DAM] Fetched {len(series)} rows; latest timestamp: {series.index.max()}")

    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    df = series.to_frame()
    df.index = df.index.tz_convert("UTC")
    local_path = LOCAL_TMP_DIR / "dam_prices_historical.csv"
    df.to_csv(local_path)
    print(f"[DAM] Saved locally to {local_path}")

    blob_name = _historical_blob_name("dam_prices", HISTORY_START, end_label)
    url = upload_local_to_blob(local_path, STORAGE_ACCOUNT, CONTAINER, blob_name)
    print(f"[DAM] Uploaded → {url}")


def backfill_load_forecast(today: pd.Timestamp) -> None:
    """Fetch and upload load forecast from HISTORY_START through end-of-tomorrow."""
    end = today + pd.Timedelta(days=2)
    print(f"[LOAD] Fetching {HISTORY_START.date()} → {(today + pd.Timedelta(days=1)).date()}")
    series = fetch_load_forecast(start=HISTORY_START, end=end, country_code=COUNTRY_CODE)
    print(f"[LOAD] Fetched {len(series)} rows; latest timestamp: {series.index.max()}")
    end_label = pd.Timestamp(series.index.max()).tz_convert(TIMEZONE).normalize()

    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    df = series.to_frame()
    df.index = df.index.tz_convert("UTC")
    local_path = LOCAL_TMP_DIR / "load_forecast_historical.csv"
    df.to_csv(local_path)
    print(f"[LOAD] Saved locally to {local_path}")

    blob_name = _historical_blob_name("load_forecast", HISTORY_START, end_label)
    url = upload_local_to_blob(local_path, STORAGE_ACCOUNT, CONTAINER, blob_name)
    print(f"[LOAD] Uploaded → {url}")


def backfill_renewable_forecast(today: pd.Timestamp) -> None:
    """Fetch and upload wind+solar forecast from HISTORY_START through end-of-tomorrow."""
    end = today + pd.Timedelta(days=2)
    print(f"[RENEW] Fetching {HISTORY_START.date()} → {(today + pd.Timedelta(days=1)).date()}")
    df = fetch_renewable_forecast(start=HISTORY_START, end=end, country_code=COUNTRY_CODE)
    print(f"[RENEW] Fetched {len(df)} rows; latest timestamp: {df.index.max()}; columns: {list(df.columns)}")
    end_label = pd.Timestamp(df.index.max()).tz_convert(TIMEZONE).normalize()

    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.index = df.index.tz_convert("UTC")
    local_path = LOCAL_TMP_DIR / "renewable_forecast_historical.csv"
    df.to_csv(local_path)
    print(f"[RENEW] Saved locally to {local_path}")

    blob_name = _historical_blob_name("renewable_forecast", HISTORY_START, end_label)
    url = upload_local_to_blob(local_path, STORAGE_ACCOUNT, CONTAINER, blob_name)
    print(f"[RENEW] Uploaded → {url}")


def main() -> None:
    today = pd.Timestamp.now(tz=TIMEZONE).normalize()
    print(f"Backfill starting at {datetime.now()} (Athens time today: {today.date()})")
    print(f"Storage: {STORAGE_ACCOUNT}/{CONTAINER}")
    print()

    backfill_dam_prices(today)
    print()
    backfill_load_forecast(today)
    print()
    backfill_renewable_forecast(today)
    print()
    print("Backfill complete.")


if __name__ == "__main__":
    main()