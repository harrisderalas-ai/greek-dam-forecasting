"""
Training entry point for Azure ML.

Trains per-horizon LightGBM models on the curated DAM dataset, logs
metrics and parameters to MLflow, and saves the trained model bundle
as an MLflow artifact.

Designed to run as an Azure ML job. Can also be invoked locally for
testing if you have access to the data and Azure credentials.

Usage (local test):
    python -m scripts.train_azureml \\
        --dam-path data/processed/backfill_tmp/dam_prices_hourly.csv \\
        --load-path data/processed/backfill_tmp/load_forecast_hourly.csv \\
        --renewable-path data/processed/backfill_tmp/renewable_forecast_hourly.csv \\
        --output-dir outputs/

Usage (Azure ML, automatically passed by job manifest):
    Inputs come from datastore mount paths; output_dir is provided by Azure ML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.dataset import assemble_dataset
from src.train import train_per_horizon_models, predict_next_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Greek DAM forecasting model")
    parser.add_argument(
        "--dam-path",
        type=str,
        required=True,
        help="Path to dam_prices full.csv (read from datastore)",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to load_forecast full.csv",
    )
    parser.add_argument(
        "--renewable-path",
        type=str,
        required=True,
        help="Path to renewable_forecast full.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (Azure ML provides this; locally specify any dir)",
    )
    parser.add_argument(
        "--gate-closure-hour",
        type=int,
        default=12,
        help="Hour at which the DAM auction gate closes",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Number of days held out for testing",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="LightGBM n_estimators per horizon",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning_rate",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=63,
        help="LightGBM num_leaves",
    )
    return parser.parse_args()


def load_data(dam_path: str, load_path: str, renewable_path: str) -> tuple[pd.Series, pd.DataFrame]:
    """Load and assemble the curated dataset from local CSV paths."""
    print(f"[data] Reading DAM:        {dam_path}")
    dam = pd.read_csv(dam_path, index_col=0)
    dam.index = pd.to_datetime(dam.index, utc=True)
    print(f"        {len(dam)} rows, range {dam.index.min()} -> {dam.index.max()}")

    print(f"[data] Reading load:       {load_path}")
    load = pd.read_csv(load_path, index_col=0)
    load.index = pd.to_datetime(load.index, utc=True)
    print(f"        {len(load)} rows, range {load.index.min()} -> {load.index.max()}")

    print(f"[data] Reading renewable:  {renewable_path}")
    renewable = pd.read_csv(renewable_path, index_col=0)
    renewable.index = pd.to_datetime(renewable.index, utc=True)
    print(f"        {len(renewable)} rows, range {renewable.index.min()} -> {renewable.index.max()}")

    prices, exog = assemble_dataset(dam, load, renewable, join="inner")
    print(f"[data] Combined: {len(prices)} aligned rows")
    print(f"        Range: {prices.index.min()} -> {prices.index.max()}")
    print(f"        Exog columns: {list(exog.columns)}")

    return prices, exog


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow auto-tracks on Azure ML; locally it logs to ./mlruns
    print("[mlflow] Starting tracking...")

    with mlflow.start_run() as run:
        # Log all hyperparameters
        mlflow.log_params({
            "gate_closure_hour": args.gate_closure_hour,
            "test_days": args.test_days,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "horizons": "tuple(range(0, 24))",
            "same_hour_lag_days": "(1, 2, 7)",
            "context_window": 1,
        })

        # Load data
        prices, exog = load_data(args.dam_path, args.load_path, args.renewable_path)
        mlflow.log_metric("training_rows", len(prices))
        mlflow.log_metric("price_min", prices.min())
        mlflow.log_metric("price_max", prices.max())
        mlflow.log_metric("price_mean", prices.mean())

        # Train
        print("[train] Training 24 per-horizon LightGBM models...")
        lgbm_params = {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1,
        }

        result = train_per_horizon_models(
            prices,
            exog=exog,
            gate_closure_hour=args.gate_closure_hour,
            horizons=tuple(range(0, 24)),
            same_hour_lag_days=(1, 2, 7),
            context_window=1,
            test_days=args.test_days,
            lgbm_params=lgbm_params,
        )

        # Log overall metrics
        mlflow.log_metric("test_mae_overall", result.overall_test_mae)
        mlflow.log_metric("test_rmse_overall", result.overall_test_rmse)

        # Log per-horizon metrics
        print("[train] Logging per-horizon metrics...")
        for _, row in result.metrics_per_horizon.iterrows():
            h = int(row["horizon"])
            mlflow.log_metric(f"test_mae_h{h:02d}", float(row["mae"]))
            mlflow.log_metric(f"test_rmse_h{h:02d}", float(row["rmse"]))

        # Save model artifacts
        print("[save] Saving model bundle to output dir...")
        import pickle

        model_path = output_dir / "model_bundle.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(result, f)

        # Save metrics summary as JSON for downstream inspection
        metrics_path = output_dir / "metrics_summary.json"
        summary = {
            "overall_test_mae": float(result.overall_test_mae),
            "overall_test_rmse": float(result.overall_test_rmse),
            "metrics_per_horizon": result.metrics_per_horizon.to_dict("records"),
            "feature_names": result.feature_names,
            "gate_closure_hour": result.gate_closure_hour,
            "horizons": list(result.horizons),
            "same_hour_lag_days": list(result.same_hour_lag_days),
        }
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Log artifacts to MLflow (these get tracked alongside the run)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        mlflow.log_dict(
            {
                "data_start": str(prices.index.min()),
                "data_end": str(prices.index.max()),
                "training_rows": len(prices),
                "test_days": args.test_days,
            },
            "data_lineage.json",
        )

        # Log feature importances of one horizon (h=12, midday)
        import pandas as pd
        importance = pd.DataFrame({
            "feature": result.feature_names,
            "importance": result.models[12].feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(output_dir / "feature_importance_h12.csv", index=False)
        mlflow.log_artifact(str(output_dir / "feature_importance_h12.csv"))

        # Register the model in the workspace's Model Registry
        print("[register] Registering model in workspace...")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name="greek-dam-forecaster",
        )
        print(f"[register] Registered as 'greek-dam-forecaster'")


        print("[done]")
        print(result.summary())


if __name__ == "__main__":
    main()