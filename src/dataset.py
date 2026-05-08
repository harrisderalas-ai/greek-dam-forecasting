"""
Assemble the curated ML training dataset from processed sources.

Bronze (raw/) → Silver (processed/) → Gold (curated/).

This module joins three processed sources into the format expected by
`build_supervised_dataset`: a prices Series (target) and an exog DataFrame
(features), aligned on timestamp.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.data_loader import read_csv_from_blob, upload_dataframe_to_blob


# Column name conventions — single source of truth
PRICE_COL = "price_eur_mwh"
LOAD_COL = "load_forecast_mw"
SOLAR_COL = "solar"
WIND_COL = "wind_onshore"
EXOG_COLS = (LOAD_COL, SOLAR_COL, WIND_COL)


def assemble_dataset(
    dam: pd.DataFrame,
    load: pd.DataFrame,
    renewable: pd.DataFrame,
    join: Literal["inner", "outer"] = "inner",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Join three processed-tier DataFrames into a (prices, exog) pair.

    Pure function. No I/O. Suitable for unit testing.

    Parameters
    ----------
    dam : DataFrame with one column (DAM prices in EUR/MWh)
    load : DataFrame with one column (load forecast in MW)
    renewable : DataFrame with two columns (solar, wind_onshore in MW)
    join : 'inner' keeps only timestamps with all three sources populated;
           'outer' keeps all timestamps and pads with NaN where any are missing.
           Default 'inner' is right for training; 'outer' is right for inference.

    Returns
    -------
    prices : pd.Series
        DAM prices indexed by timestamp.
    exog : pd.DataFrame
        Exogenous features (load_forecast_mw, solar, wind_onshore).
        Same index as `prices`.

    Raises
    ------
    ValueError
        If any input has the wrong number of columns or the wrong column names.
    """
    # Validate input shapes
    _validate_dam(dam)
    _validate_load(load)
    _validate_renewable(renewable)

    # Concat all on column axis, aligning by index
    combined = pd.concat([dam, load, renewable], axis=1, join=join).sort_index()

    # The first column is always price (DAM); the rest are exog
    prices = combined[PRICE_COL].rename("price_eur_mwh")
    exog = combined[list(EXOG_COLS)]

    return prices, exog


def load_combined_dataset(
    storage_account: str,
    container: str = "processed",
    join: Literal["inner", "outer"] = "inner",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Read the three processed full.csv files from blob and assemble them.

    Convenience wrapper around `assemble_dataset` for the production case
    of reading from Azure Blob Storage.
    """
    dam = read_csv_from_blob(storage_account, container, "dam_prices/full.csv")
    load = read_csv_from_blob(storage_account, container, "load_forecast/full.csv")
    renewable = read_csv_from_blob(
        storage_account, container, "renewable_forecast/full.csv"
    )
    return assemble_dataset(dam, load, renewable, join=join)


def save_curated_dataset(
    storage_account: str,
    container: str = "curated",
    join: Literal["inner", "outer"] = "inner",
) -> tuple[str, str]:
    """
    Build the curated dataset from processed/, save to curated/.

    Reads:  processed/{dam_prices,load_forecast,renewable_forecast}/full.csv
    Writes: curated/training_dataset.csv (prices + exog joined, single file)

    Returns the URLs of the written blobs (training and meta).
    """
    prices, exog = load_combined_dataset(
        storage_account, container="processed", join=join
    )

    # Combined into one DataFrame for storage
    combined = pd.concat([prices.rename(PRICE_COL), exog], axis=1)

    blob_name = "training_dataset.csv"
    url = upload_dataframe_to_blob(combined, storage_account, container, blob_name)

    return url, blob_name


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_dam(df: pd.DataFrame) -> None:
    if df.shape[1] != 1:
        raise ValueError(f"DAM must have 1 column, got {df.shape[1]}")
    if df.columns[0] != PRICE_COL:
        raise ValueError(f"DAM column must be '{PRICE_COL}', got '{df.columns[0]}'")


def _validate_load(df: pd.DataFrame) -> None:
    if df.shape[1] != 1:
        raise ValueError(f"Load must have 1 column, got {df.shape[1]}")
    if df.columns[0] != LOAD_COL:
        raise ValueError(f"Load column must be '{LOAD_COL}', got '{df.columns[0]}'")


def _validate_renewable(df: pd.DataFrame) -> None:
    if df.shape[1] != 2:
        raise ValueError(f"Renewable must have 2 columns, got {df.shape[1]}")
    if not all(c in df.columns for c in (SOLAR_COL, WIND_COL)):
        raise ValueError(
            f"Renewable must have columns {SOLAR_COL!r} and {WIND_COL!r}; "
            f"got {list(df.columns)}"
        )