"""
Daily orchestrator for the Greek DAM forecasting pipeline.

Triggered daily by a timer. Runs the full pipeline:
1. Fetch ENTSO-E data
2. Process raw → processed
3. Submit Azure ML training job
4. Wait for completion
5. Submit Azure ML inference job
6. Wait for completion
"""

import logging
import os
import sys
import time
from pathlib import Path

import azure.functions as func
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

# Make our scripts importable
# When deployed, the Function App's working dir contains our code
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import fetch_daily, process_raw_to_processed

# Configuration — these would come from app settings in production
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "rg-greekdam-dev-westeu")
WORKSPACE = os.environ.get("AZURE_WORKSPACE", "mlw-greekdam-dev-westeu")

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 8 * * *",            # cron: at 08:00 UTC daily           # cron: at 08:00 UTC daily  
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
def daily_pipeline(timer: func.TimerRequest) -> None:
    """Daily MLOps pipeline: fetch → process → train → predict."""
    
    logging.info("=" * 60)
    logging.info("Daily DAM forecasting pipeline starting")
    logging.info("=" * 60)
    
    try:
        # Phase 1: Fetch latest data
        logging.info("\n[Phase 1] Fetching ENTSO-E data...")
        fetch_daily.main()  # Uses default run_date = today

        # Phase 2: Process raw → processed
        logging.info("\n[Phase 2] Processing raw → processed...")
        process_raw_to_processed.main()

        # Phase 3 + 4: Submit and wait for training + inference
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE,
        )

        logging.info("\n[Phase 3] Submitting Azure ML training job...")
        training_job = submit_job(ml_client, "./infra/training_job.yml", "train")
        wait_for_job(ml_client, training_job.name, "training")

        logging.info("\n[Phase 4] Submitting Azure ML inference job...")
        # Update inference job to point at tomorrow
        from datetime import datetime, timedelta
        tomorrow_athens = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        inference_job = submit_inference_job(ml_client, tomorrow_athens)
        wait_for_job(ml_client, inference_job.name, "inference")

        logging.info("\n" + "=" * 60)
        logging.info("Pipeline completed successfully")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


def submit_job(ml_client: MLClient, job_yaml_path: str, job_type: str) -> Job:
    """Submit a job from a YAML file."""
    from azure.ai.ml import load_job

    job = load_job(source=str(Path(__file__).parent / job_yaml_path))
    
    # Force unique name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job.name = f"{job_type}-{timestamp}"
    
    submitted = ml_client.jobs.create_or_update(job)
    logging.info(f"  Submitted job: {submitted.name}")
    logging.info(f"  Studio URL: {submitted.studio_url}")
    return submitted


def submit_inference_job(ml_client: MLClient, target_date: str) -> Job:
    """Submit inference job with the target date as an override."""
    from azure.ai.ml import load_job

    job = load_job(source=str(Path(__file__).parent / "infra/inference_job.yml"))
    
    # Override the target-date-athens argument
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job.name = f"inference-{timestamp}"
    
    # The command in the YAML has `--target-date-athens 2026-05-14` hardcoded.
    # We need to override it at submission time.
    # The easiest way: use job.command directly
    job.command = job.command.replace(
        "--target-date-athens 2026-05-14",
        f"--target-date-athens {target_date}",
    )
    
    submitted = ml_client.jobs.create_or_update(job)
    logging.info(f"  Submitted job: {submitted.name}")
    logging.info(f"  Studio URL: {submitted.studio_url}")
    return submitted


def wait_for_job(ml_client: MLClient, job_name: str, label: str, timeout_minutes: int = 15) -> None:
    """Poll job status until completed or timeout."""
    start = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        job = ml_client.jobs.get(job_name)
        status = job.status
        
        elapsed = int(time.time() - start)
        logging.info(f"  [{label}] {job_name}: {status} ({elapsed}s elapsed)")
        
        if status in ("Completed", "Failed", "Canceled"):
            if status != "Completed":
                raise RuntimeError(f"{label} job failed with status: {status}")
            return
        
        if elapsed > timeout_seconds:
            raise TimeoutError(f"{label} job did not complete in {timeout_minutes} minutes")
        
        time.sleep(15)