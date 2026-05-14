"""
Fetch the latest ENTSO-E data and upload to blob storage.

Runs once per day (typically at 08:00 Athens). Fetches a window covering
yesterday through tomorrow for DAM prices, load forecast, and renewable
forecast. Uploads each to raw/{kind}/daily/{YYYY-MM-DD}.csv.

The date in the filename is the run date (Athens local). The data inside
spans yesterday + today.

Usage:
    python -m scripts.fetch_daily
    python -m scripts.fetch_daily --run-date 2026-05-14
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from src.data_loader import (
    fetch_dam_prices,
    fetch_load_forecast,
    fetch_renewable_forecast,
)


STORAGE_ACCOUNT = "sagreekdamdevweu"
CONTAINER = "raw"
COUNTRY_CODE = "GR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily ENTSO-E data")
    parser.add_argument(
        "--run-date",
        default=None,
        help="Date the fetch represents in Athens (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Days back from run-date to start fetch (default: 1 = yesterday)",
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=1,
        help="Days forward from run-date to end fetch (default: 1 = tomorrow)",
    )
    return parser.parse_args()


def upload_csv_to_blob(df: pd.DataFrame, blob_path: str) -> None:
    """Upload a DataFrame as CSV to blob storage."""
    print(f"  -> Uploading to {CONTAINER}/{blob_path}")
    credential = DefaultAzureCredential()
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=CONTAINER, blob=blob_path)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
    print(f"  [OK] {len(df)} rows uploaded")


def main(
    run_date: str | None = None,
    days_back: int = 1,
    days_forward: int = 1,
) -> None:
    """
    Fetch ENTSO-E data and upload to blob storage.

    Args:
        run_date: Date the fetch represents in Athens (YYYY-MM-DD).
                  If None, defaults to today (Athens).
        days_back: Days back from run_date to start the fetch window.
        days_forward: Days forward from run_date to end the fetch window.
    """
    # Determine the run date (Athens local)
    if run_date:
        run_date_athens = pd.Timestamp(run_date, tz="Europe/Athens")
    else:
        now_athens = pd.Timestamp.now(tz="Europe/Athens")
        run_date_athens = now_athens.normalize()

    # Fetch window
    fetch_start_athens = run_date_athens - pd.Timedelta(days=days_back)
    fetch_end_athens = run_date_athens + pd.Timedelta(days=days_forward + 1)

    run_date_str = run_date_athens.strftime("%Y-%m-%d")

    print(f"=== Daily fetch for run-date {run_date_str} ===")
    print(f"Window (Athens): {fetch_start_athens} -> {fetch_end_athens}")
    print(f"                = {fetch_start_athens.tz_convert('UTC')} -> {fetch_end_athens.tz_convert('UTC')} UTC")
    print()

    # 1. DAM prices
    print(f"[1/3] Fetching DAM prices...")
    try:
        dam = fetch_dam_prices(
            start=fetch_start_athens,
            end=fetch_end_athens,
            country_code=COUNTRY_CODE,
        )
        # Convert to UTC for consistency with processed data
        dam.index = dam.index.tz_convert("UTC")
        print(f"  -> {len(dam)} rows, {dam.index.min()} -> {dam.index.max()}")
        blob_path = f"dam_prices/daily/{run_date_str}.csv"
        upload_csv_to_blob(dam.to_frame(name="price_eur_mwh"), blob_path)
    except Exception as e:
        print(f"  [FAIL] DAM fetch failed: {e}")
        # Don't fail the whole script if DAM is unavailable; log and continue
        # (sometimes ENTSO-E is slow)

    print()

    # 2. Load forecast
    print(f"[2/3] Fetching load forecast...")
    try:
        load = fetch_load_forecast(
            start=fetch_start_athens,
            end=fetch_end_athens,
            country_code=COUNTRY_CODE,
        )
        load.index = load.index.tz_convert("UTC")
        print(f"  -> {len(load)} rows, {load.index.min()} -> {load.index.max()}")
        blob_path = f"load_forecast/daily/{run_date_str}.csv"
        upload_csv_to_blob(load.to_frame(name="load_forecast_mw"), blob_path)
    except Exception as e:
        print(f"  [FAIL] Load forecast fetch failed: {e}")

    print()

    # 3. Renewable forecast
    print(f"[3/3] Fetching renewable forecast...")
    try:
        renewable = fetch_renewable_forecast(
            start=fetch_start_athens,
            end=fetch_end_athens,
            country_code=COUNTRY_CODE,
        )
        renewable.index = renewable.index.tz_convert("UTC")
        print(f"  -> {len(renewable)} rows, {renewable.index.min()} -> {renewable.index.max()}")
        blob_path = f"renewable_forecast/daily/{run_date_str}.csv"
        upload_csv_to_blob(renewable, blob_path)
    except Exception as e:
        print(f"  [FAIL] Renewable forecast fetch failed: {e}")

    print()
    print(f"=== Daily fetch complete ===")


if __name__ == "__main__":
    args = parse_args()
    main(
        run_date=args.run_date,
        days_back=args.days_back,
        days_forward=args.days_forward,
    )