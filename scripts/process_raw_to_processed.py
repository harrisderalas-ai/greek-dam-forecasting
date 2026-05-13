"""
Read all raw CSVs from Blob Storage (historical + daily increments),
resample each to hourly, concatenate, de-duplicate, sort, and write to processed/.

Each file is resampled to hourly before concatenation. This handles
sources that publish at different cadences (e.g., DAM moved to 15-min in Oct 2025).

Source of truth: Blob Storage.
Output: one canonical "full.csv" per kind, in processed/{kind}/full.csv.
This file is overwritten each run.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import (
    list_blobs_in_prefix,
    read_csv_from_blob,
    upload_dataframe_to_blob,
)
from src.processing import process_raw_dataframe

STORAGE_ACCOUNT = "sagreekdamdevweu"
DATA_KINDS = ("dam_prices", "load_forecast", "renewable_forecast")


def list_all_raw_blobs_for_kind(kind: str) -> list[str]:
    """
    Return all raw blob names contributing to this kind's history.

    Looks in:
      raw/{kind}/historical_*.csv  (the backfill)
      raw/{kind}/daily/*.csv        (daily increments)
    """
    historical = list_blobs_in_prefix(STORAGE_ACCOUNT, "raw", f"{kind}/historical_")
    daily = list_blobs_in_prefix(STORAGE_ACCOUNT, "raw", f"{kind}/daily/")
    all_blobs = [b for b in historical + daily if b.endswith(".csv")]
    return sorted(all_blobs)


def read_and_concatenate(blob_names: list[str]) -> pd.DataFrame:
    """
    Read each blob, resample to hourly individually, concatenate, dedupe.
    
    Resampling each file independently before concatenation ensures the
    dedup step compares apples to apples (all timestamps are hourly).
    """
    if not blob_names:
        raise ValueError("No blobs to read")

    frames = []
    for name in blob_names:
        df = read_csv_from_blob(STORAGE_ACCOUNT, "raw", name)
        df_hourly, _ = process_raw_dataframe(df)
        frames.append(df_hourly)
        print(f"    {name}: {len(df)} raw -> {len(df_hourly)} hourly")

    combined = pd.concat(frames, axis=0)
    n_before = len(combined)
    combined = combined[~combined.index.duplicated(keep="last")]
    n_after = len(combined)
    if n_before != n_after:
        print(f"    Deduplicated: {n_before} -> {n_after} rows ({n_before - n_after} duplicates)")

    return combined.sort_index()


def process_one_kind(kind: str) -> None:
    """Build the full hourly history for `kind` and upload as full.csv."""
    print(f"\n=== {kind} ===")

    raw_blobs = list_all_raw_blobs_for_kind(kind)
    if not raw_blobs:
        print(f"  No raw blobs found for {kind}; skipping")
        return

    print(f"  Found {len(raw_blobs)} raw blob(s):")
    combined = read_and_concatenate(raw_blobs)
    print(
        f"  Combined: {len(combined)} rows, "
        f"range {combined.index.min()} -> {combined.index.max()}"
    )

    # Already hourly from the per-file resampling, but ensure consistency
    # (handles any cross-file ordering subtleties)
    processed_df, report = process_raw_dataframe(combined)
    print(f"  Process report: {report}")
    print(f"  Final range: {processed_df.index.min()} -> {processed_df.index.max()}")
    print(f"  Final rows:  {len(processed_df)}")

    out_blob = f"{kind}/full.csv"
    print(f"  Writing: {out_blob}")
    url = upload_dataframe_to_blob(
        processed_df, STORAGE_ACCOUNT, "processed", out_blob
    )
    print(f"  -> {url}")


def main() -> None:
    print(f"Processing raw -> processed for {STORAGE_ACCOUNT}")
    for kind in DATA_KINDS:
        process_one_kind(kind)
    print("\nDone.")


if __name__ == "__main__":
    main()