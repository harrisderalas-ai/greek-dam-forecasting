"""
Predict tomorrow's 24 hourly DAM prices using the latest registered model.

Loads the latest version of the 'greek-dam-forecaster' model from the
Azure ML Model Registry, fetches the current processed dataset, builds
features for tomorrow's 24 horizons, and saves predictions to blob.

Designed to run as an Azure ML job. Output is written to a folder
that gets uploaded to the curated/ container.

Usage:
    python -m scripts.predict_tomorrow \\
        --dam-path <path-to-dam-csv> \\
        --load-path <path-to-load-csv> \\
        --renewable-path <path-to-renewable-csv> \\
        --output-dir <where-to-save-predictions> \\
        --gate-closure-hour 12 \\
        --model-name greek-dam-forecaster
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from src.dataset import assemble_dataset
from src.train import predict_next_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tomorrow's DAM prices")
    parser.add_argument("--dam-path", required=True)
    parser.add_argument("--load-path", required=True)
    parser.add_argument("--renewable-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-closure-hour", type=int, default=12)
    parser.add_argument(
        "--model-name",
        default="greek-dam-forecaster",
        help="Registered model name",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Specific model version. If omitted, uses the latest version.",
    )
    return parser.parse_args()


def get_workspace_info_from_env() -> dict:
    """
    Get workspace info from environment variables (set by Azure ML at runtime).
    """
    import os
    return {
        "subscription_id": os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
        "resource_group_name": os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
        "workspace_name": os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
    }


def download_latest_model(model_name: str, version: str | None) -> str:
    """
    Download the model artifact from the registry.
    Returns the local path to the downloaded folder.
    """
    workspace_info = get_workspace_info_from_env()

    if not all(workspace_info.values()):
        # Running locally — workspace info must be provided differently
        # For now, raise if not on Azure ML
        raise RuntimeError(
            "Workspace info not in environment. "
            "Use az ml job to run this on Azure ML compute."
        )

    print(f"[model] Connecting to workspace: {workspace_info['workspace_name']}")
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential=credential, **workspace_info)

    # Get the specified version or the latest
    if version:
        model = ml_client.models.get(name=model_name, version=version)
        print(f"[model] Using specified version {version}")
    else:
        # Find latest version by sorting all versions
        all_versions = list(ml_client.models.list(name=model_name))
        if not all_versions:
            raise RuntimeError(f"No versions found for model '{model_name}'")
        # Sort by version (treat as int); take the highest
        all_versions.sort(key=lambda m: int(m.version), reverse=True)
        model = all_versions[0]
        print(f"[model] Using latest version: {model.version}")

    # Download to a local folder
    print(f"[model] Downloading model artifacts...")
    download_path = Path("./downloaded_model")
    download_path.mkdir(exist_ok=True)
    ml_client.models.download(
        name=model_name,
        version=model.version,
        download_path=str(download_path),
    )
    print(f"[model] Downloaded to {download_path}")

    return str(download_path)


def load_model_bundle(model_path: str):
    """Find and unpickle the trained model bundle."""
    # The pickle is buried in the downloaded folder; search for it
    path = Path(model_path)
    pkl_files = list(path.rglob("model_bundle.pkl"))
    if not pkl_files:
        raise RuntimeError(f"No model_bundle.pkl found in {path}")

    pkl_path = pkl_files[0]
    print(f"[model] Loading bundle from {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_data(dam_path: str, load_path: str, renewable_path: str) -> tuple[pd.Series, pd.DataFrame]:
    """Load processed CSVs and assemble (with outer join so tomorrow's exog is included)."""
    print(f"[data] Reading DAM from {dam_path}")
    dam = pd.read_csv(dam_path, index_col=0)
    dam.index = pd.to_datetime(dam.index, utc=True)

    print(f"[data] Reading load from {load_path}")
    load = pd.read_csv(load_path, index_col=0)
    load.index = pd.to_datetime(load.index, utc=True)

    print(f"[data] Reading renewable from {renewable_path}")
    renewable = pd.read_csv(renewable_path, index_col=0)
    renewable.index = pd.to_datetime(renewable.index, utc=True)

    # Outer join: keeps timestamps where exog exists but DAM is unknown
    prices, exog = assemble_dataset(dam, load, renewable, join="outer")
    print(f"[data] Combined: {len(prices)} rows")
    print(f"        DAM range: {prices.dropna().index.min()} → {prices.dropna().index.max()}")
    print(f"        Exog range: {exog.dropna().index.min()} → {exog.dropna().index.max()}")
    return prices, exog


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        # 1. Download the latest registered model
        model_dir = download_latest_model(args.model_name, args.model_version)
        result = load_model_bundle(model_dir)
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("model_version", args.model_version or "latest")

        # 2. Load the data
        prices, exog = load_data(args.dam_path, args.load_path, args.renewable_path)

        # 3. Predict tomorrow's 24 prices
        print(f"[predict] Generating forecast for next day...")
        forecast = predict_next_day(result, prices, exog=exog)

        # forecast is a pd.Series indexed by tomorrow's 24 hours
        print(f"[predict] Generated {len(forecast)} predictions")
        print(f"        Range: {forecast.index.min()} → {forecast.index.max()}")
        print(f"        Mean: {forecast.mean():.2f}, Min: {forecast.min():.2f}, Max: {forecast.max():.2f}")

        # 4. Save
        # Forecast date is tomorrow's date in Athens
        athens_tomorrow = forecast.index.min().tz_convert("Europe/Athens").date()
        forecast_filename = f"{athens_tomorrow.strftime('%Y-%m-%d')}_forecast.csv"
        output_path = output_dir / forecast_filename

        forecast_df = forecast.to_frame(name="predicted_price_eur_mwh")
        forecast_df.index = forecast_df.index.tz_convert("UTC")
        forecast_df.to_csv(output_path)
        print(f"[save] Saved forecast to {output_path}")

        # 5. Log to MLflow
        mlflow.log_metric("forecast_mean", float(forecast.mean()))
        mlflow.log_metric("forecast_min", float(forecast.min()))
        mlflow.log_metric("forecast_max", float(forecast.max()))
        mlflow.log_artifact(str(output_path), artifact_path="forecast")

        # 6. Optionally: write a metadata file
        meta = {
            "model_name": args.model_name,
            "model_version": args.model_version or "latest",
            "forecast_date_athens": athens_tomorrow.isoformat(),
            "forecast_hours_utc": [str(t) for t in forecast.index],
            "predicted_prices_eur_mwh": [float(v) for v in forecast.values],
        }
        meta_path = output_dir / f"{athens_tomorrow.strftime('%Y-%m-%d')}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] Saved metadata to {meta_path}")
        mlflow.log_artifact(str(meta_path), artifact_path="forecast")

        print("[done] Inference complete")


if __name__ == "__main__":
    main()