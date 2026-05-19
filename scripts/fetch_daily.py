"""
Fetch the latest ENTSO-E data and upload to blob storage.

Runs once per day. Fetches data for DAM prices, load forecast, and renewable
forecast. Uploads each to raw/{kind}/daily/{YYYY-MM-DD}.csv.

The date in the filename is the run date (Athens local).

DAM fetch window:    [run_date - days_back, run_date + 1)     — published past + today
Load/Renewable:      [run_date - days_back, run_date + days_forward + 1)

Why different windows?
At gate closure today (12:00 Athens), the DAM auction for TOMORROW is happening.
Tomorrow's prices are NOT yet published. But TODAY's prices WERE published
yesterday (after yesterday's auction). So we fetch through end-of-TODAY for DAM.

Load and renewable forecasts publish day-ahead, so we go one more day forward
(tomorrow's load and renewable forecasts are already published).

Usage:
    python -m scripts.fetch_daily
    python -m scripts.fetch_daily --run-date 2026-05-14
"""

from __future__ import annotations

import argparse
import io
import logging

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

logger = logging.getLogger(__name__)


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
        help="Days forward from run-date to end fetch for load/renewable (default: 1 = tomorrow)",
    )
    return parser.parse_args()


def upload_csv_to_blob(df: pd.DataFrame, blob_path: str) -> None:
    """Upload a DataFrame as CSV to blob storage."""
    logger.info(f"  -> Uploading to {CONTAINER}/{blob_path}")
    credential = DefaultAzureCredential()
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service.get_blob_client(container=CONTAINER, blob=blob_path)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
    logger.info(f"  [OK] {len(df)} rows uploaded")


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
        days_forward: Days forward from run_date to end the fetch window
                      for load/renewable forecasts. (DAM always ends at
                      end-of-today; tomorrow's prices aren't published yet.)
    """
    # Determine the run date (Athens local)
    if run_date:
        run_date_athens = pd.Timestamp(run_date, tz="Europe/Athens")
    else:
        now_athens = pd.Timestamp.now(tz="Europe/Athens")
        run_date_athens = now_athens.normalize()

    # Common fetch start (yesterday by default)
    fetch_start_athens = run_date_athens - pd.Timedelta(days=days_back)

    # Two different end points:
    #   DAM:              ends at run_date + 1 day (exclusive) — includes today.
    #                     Today's prices were published yesterday afternoon by
    #                     the DAM auction, so they're publicly available now.
    #                     Tomorrow's prices are NOT yet published (the auction
    #                     for tomorrow closes around noon today).
    #   Load/renewable:   ends at run_date + days_forward + 1 (includes tomorrow).
    #                     These forecasts publish day-ahead.
    dam_fetch_end = run_date_athens + pd.Timedelta(days=1)
    forecast_fetch_end = run_date_athens + pd.Timedelta(days=days_forward + 1)

    run_date_str = run_date_athens.strftime("%Y-%m-%d")

    logger.info(f"=== Daily fetch for run-date {run_date_str} ===")
    logger.info(f"DAM window (Athens):       {fetch_start_athens} -> {dam_fetch_end}")
    logger.info(f"Forecast window (Athens):  {fetch_start_athens} -> {forecast_fetch_end}")

    # 1. DAM prices
    logger.info("[1/3] Fetching DAM prices...")
    try:
        dam = fetch_dam_prices(
            start=fetch_start_athens,
            end=dam_fetch_end,
            country_code=COUNTRY_CODE,
        )
        # Convert to UTC for consistency with processed data
        dam.index = dam.index.tz_convert("UTC")
        logger.info(f"  -> {len(dam)} rows, {dam.index.min()} -> {dam.index.max()}")
        blob_path = f"dam_prices/daily/{run_date_str}.csv"
        upload_csv_to_blob(dam.to_frame(name="price_eur_mwh"), blob_path)
    except Exception as e:
        logger.exception(f"  [FAIL] DAM fetch failed: {e}")
        # Don't fail the whole script if DAM is unavailable; log and continue
        # (sometimes ENTSO-E is slow)

    # 2. Load forecast (publishes day-ahead — tomorrow's forecast IS available)
    logger.info("[2/3] Fetching load forecast...")
    try:
        load = fetch_load_forecast(
            start=fetch_start_athens,
            end=forecast_fetch_end,
            country_code=COUNTRY_CODE,
        )
        load.index = load.index.tz_convert("UTC")
        logger.info(f"  -> {len(load)} rows, {load.index.min()} -> {load.index.max()}")
        blob_path = f"load_forecast/daily/{run_date_str}.csv"
        upload_csv_to_blob(load.to_frame(name="load_forecast_mw"), blob_path)
    except Exception as e:
        logger.exception(f"  [FAIL] Load forecast fetch failed: {e}")

    # 3. Renewable forecast (publishes day-ahead — tomorrow's forecast IS available)
    logger.info("[3/3] Fetching renewable forecast...")
    try:
        renewable = fetch_renewable_forecast(
            start=fetch_start_athens,
            end=forecast_fetch_end,
            country_code=COUNTRY_CODE,
        )
        renewable.index = renewable.index.tz_convert("UTC")
        logger.info(
            f"  -> {len(renewable)} rows, {renewable.index.min()} -> {renewable.index.max()}"
        )
        blob_path = f"renewable_forecast/daily/{run_date_str}.csv"
        upload_csv_to_blob(renewable, blob_path)
    except Exception as e:
        logger.exception(f"  [FAIL] Renewable forecast fetch failed: {e}")

    logger.info("=== Daily fetch complete ===")


if __name__ == "__main__":
    # When run as a CLI script, configure basic logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # match the previous bare-print style
    )
    args = parse_args()
    main(
        run_date=args.run_date,
        days_back=args.days_back,
        days_forward=args.days_forward,
    )
