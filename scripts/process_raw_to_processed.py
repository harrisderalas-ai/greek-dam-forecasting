"""
Read all raw CSVs from Blob Storage (historical + daily increments),
concatenate, de-duplicate, sort, normalize to hourly, and write to processed/.

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
    """Read each blob, concatenate, de-duplicate by index, sort by index."""
    if not blob_names:
        raise ValueError("No blobs to read")

    frames = []
    for name in blob_names:
        df = read_csv_from_blob(STORAGE_ACCOUNT, "raw", name)
        frames.append(df)
        print(f"    {name}: {len(df)} rows")

    combined = pd.concat(frames, axis=0)
    n_before = len(combined)
    combined = combined[~combined.index.duplicated(keep="last")]
    n_after = len(combined)
    if n_before != n_after:
        print(f"    Deduplicated: {n_before} -> {n_after} rows ({n_before - n_after} dupes removed)")

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

    processed_df, report = process_raw_dataframe(combined)
    print(f"  Process report: {report}")

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