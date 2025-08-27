"""
Module for running YOLO training on Google Cloud's Vertex AI with enhanced artifact handling.

This version improves artifact scanning to detect model weights (`best.pt` or `last.pt`)
correctly in GCS and supports flexible weight file detection, logging, and retry logic.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import CustomContainerTrainingJob, CustomJob

# --- Basic Configuration ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("google.cloud.aiplatform").setLevel(logging.WARNING)


class GoogleCloudTrainer:
    """Manages YOLO training jobs lifecycle on Vertex AI with robust artifact handling."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        staging_bucket: Optional[str] = None,
        service_account: Optional[str] = None,
        enable_managed_model: bool = False,
        managed_model_container_uri: Optional[str] = None,
        managed_model_display_name: Optional[str] = None,
    ):
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket or f"gs://{project_id}-yolo-training"
        self.service_account = service_account
        self.enable_managed_model = enable_managed_model
        self.managed_model_container_uri = managed_model_container_uri
        self.managed_model_display_name = managed_model_display_name

        self.gcs_client = storage.Client()

        aiplatform.init(
            project=self.project_id,
            location=self.location,
            staging_bucket=self.staging_bucket,
        )
        logger.info(f"Initialized Google Cloud trainer for project {self.project_id} in {self.location}")

    def _get_training_container_uri(self) -> str:
        """Returns full URI for the training container in Artifact Registry."""
        region = os.getenv("GCP_ARTIFACT_REGION", "us-central1")
        repo = os.getenv("GCP_ARTIFACT_REPO", "yolo-containers")
        image = os.getenv("GCP_ARTIFACT_IMAGE", "ultralytics:latest")
        default_uri = f"{region}-docker.pkg.dev/{self.project_id}/{repo}/{image}"
        return os.getenv("YOLO_CONTAINER_URI", default_uri)

    async def submit_training_job(
        self,
        dataset_yaml: str,
        advanced_params: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Submit a custom container training job to Vertex AI.

        Supports optional managed model creation by passing model serving container URI
        and display name.
        """
        try:
            container_uri = self._get_training_container_uri()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_display_name = f"yolo-training-{timestamp}"

            training_args = [
                "train",
                f"data={dataset_yaml}",
                "name=training_run",
                "device=0",
            ]

            if advanced_params:
                for key, value in advanced_params.items():
                    training_args.append(f"{key}={value}")

            job_kwargs = {
                "display_name": job_display_name,
                "container_uri": container_uri,
                "staging_bucket": self.staging_bucket,
            }

            # Include managed model creation parameters if enabled
            if self.enable_managed_model:
                if not (self.managed_model_container_uri and self.managed_model_display_name):
                    raise ValueError(
                        "Managed model parameters must be set when enable_managed_model=True."
                    )
                job_kwargs.update(
                    {
                        "model_serving_container_image_uri": self.managed_model_container_uri,
                        "model_display_name": self.managed_model_display_name,
                    }
                )

            job = CustomContainerTrainingJob(**job_kwargs)

            base_output_dir = f"{self.staging_bucket}/{job_display_name}"
            logger.info(f"✅ Job output will be saved to permanent location: {base_output_dir}")

            job.run(
                args=training_args,
                replica_count=1,
                machine_type=kwargs.get("machine_type", "n1-standard-8"),
                accelerator_type=kwargs.get("accelerator_type", "NVIDIA_TESLA_T4"),
                accelerator_count=kwargs.get("accelerator_count", 1),
                service_account=self.service_account,
                sync=False,
                base_output_dir=base_output_dir,
            )

            job._wait_for_resource_creation()

            job_info = {
                "job_id": job.name,
                "job_display_name": job.display_name,
                "status": "SUBMITTED",
            }
            return {"success": True, "job_info": job_info}

        except Exception as e:
            logger.error(f"❌ Error submitting training job: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def monitor_training_job(self, job_name: str, poll_interval: int = 30, max_wait_minutes: int = 360) -> Dict:
        """
        Poll a running Vertex AI job until it finishes, with configurable polling interval and max wait time.

        Returns job result and artifact output directory.
        """
        try:
            logger.info(f"🔍 Monitoring Vertex AI job: {job_name}")
            start_time = time.time()
            terminal_states = {
                "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                "PIPELINE_STATE_SUCCEEDED", "PIPELINE_STATE_FAILED", "PIPELINE_STATE_CANCELLED",
            }

            last_state = None

            # Try to get a CustomContainerTrainingJob or fallback to CustomJob
            job = await asyncio.to_thread(
                aiplatform.CustomContainerTrainingJob.get, resource_name=job_name
            )
            if not job:
                job = await asyncio.to_thread(aiplatform.CustomJob.get, resource_name=job_name)

            while job.state.name not in terminal_states:
                if job.state.name != last_state:
                    logger.info(f"⏳ Training status for '{job.display_name}': {job.state.name}")
                    last_state = job.state.name

                await asyncio.sleep(poll_interval)

                job = await asyncio.to_thread(
                    aiplatform.CustomContainerTrainingJob.get, resource_name=job_name
                )
                if not job:
                    job = await asyncio.to_thread(aiplatform.CustomJob.get, resource_name=job_name)

                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes > max_wait_minutes:
                    logger.error(f"⏰ Training job exceeded max wait time of {max_wait_minutes} minutes.")
                    return {"success": False, "status": "TIMEOUT", "error": "Monitoring timeout exceeded."}

            logger.info(f"Job '{job.display_name}' finished with state: {job.state.name}")

            if "SUCCEEDED" in job.state.name:
                backing_custom_job_name = None
                try:
                    backing_custom_job_name = job._gca_resource.training_task_metadata.get("backingCustomJob")
                except Exception:
                    backing_custom_job_name = None

                if backing_custom_job_name:
                    custom_job = await asyncio.to_thread(
                        aiplatform.CustomJob.get, resource_name=backing_custom_job_name
                    )
                    output_dir = custom_job.job_spec.base_output_directory.output_uri_prefix
                else:
                    output_dir = getattr(job, "base_output_dir", None) or ""

                if not output_dir:
                    logger.warning("Output directory URI could not be determined.")

                results = self._construct_result_paths(output_dir)
                return {
                    "success": True,
                    "status": "COMPLETED",
                    "output_dir": results.get("output_dir"),
                    "results": results,
                }
            else:
                error_message = (
                    job.error.message
                    if hasattr(job, "error") and job.error
                    else f"Job failed with state: {job.state.name}"
                )
                return {"success": False, "status": job.state.name, "error": error_message}

        except Exception as e:
            logger.error(f"Error monitoring job {job_name}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _construct_result_paths(self, base_gcs_output_dir: str) -> Dict:
        """
        Attempt to find artifact files (`best.pt` or `last.pt`, and `results.csv`) under the output directory in GCS,
        handling eventual consistency with retries and enhanced logging.
        """
        if not base_gcs_output_dir:
            logger.error("Base GCS output directory is empty or None.")
            return {"output_dir": None, "best_model_path": None, "results_csv_path": None}

        prefix_path = base_gcs_output_dir.replace("gs://", "").rstrip("/")

        try:
            bucket_name, prefix = prefix_path.split("/", 1)
        except Exception:
            logger.error(f"Could not parse bucket and prefix from URI: {base_gcs_output_dir}")
            return {"output_dir": base_gcs_output_dir, "best_model_path": None, "results_csv_path": None}

        bucket = self.gcs_client.bucket(bucket_name)
        max_retries = 5
        retry_delay = 20  # seconds

        search_prefix = f"{prefix}/model/training_run/"

        for attempt in range(max_retries):
            logger.info(f"🔍 Scanning for artifacts in gs://{search_prefix} (Attempt {attempt + 1}/{max_retries})")

            blobs = list(bucket.list_blobs(prefix=search_prefix))
            best_model_blob = None
            results_csv_blob = None

            for blob in blobs:
                name_lower = blob.name.lower()
                if name_lower.endswith("weights/best.pt") or name_lower.endswith("weights/last.pt"):
                    best_model_blob = blob
                elif name_lower.endswith("results.csv"):
                    results_csv_blob = blob

                if best_model_blob and results_csv_blob:
                    break

            logger.debug(
                f"Found blobs: best_model_blob={best_model_blob.name if best_model_blob else None}, "
                f"results_csv_blob={results_csv_blob.name if results_csv_blob else None}"
            )

            if best_model_blob and results_csv_blob:
                logger.info("✅ Found both best.pt/last.pt and results.csv.")
                results = {
                    "output_dir": base_gcs_output_dir,
                    "best_model_path": f"gs://{best_model_blob.bucket.name}/{best_model_blob.name}",
                    "results_csv_path": f"gs://{results_csv_blob.bucket.name}/{results_csv_blob.name}",
                }
                return results

            if attempt < max_retries - 1:
                logger.warning(f"Artifacts not yet found. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        logger.error("❌ Failed to find artifacts after multiple retries.")
        return {"output_dir": base_gcs_output_dir, "best_model_path": None, "results_csv_path": None}


# --- Singleton Factory ---
_cloud_trainer: Optional[GoogleCloudTrainer] = None


def get_cloud_trainer(enable_managed_model: bool = False) -> GoogleCloudTrainer:
    """
    Returns a thread-safe singleton instance of GoogleCloudTrainer.

    Pass enable_managed_model=True to enable managed model creation
    with required environment variables set.
    """
    global _cloud_trainer
    if _cloud_trainer is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set.")

        location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        service_account = os.environ.get("GCP_SERVICE_ACCOUNT")

        managed_model_container_uri = None
        managed_model_display_name = None
        if enable_managed_model:
            managed_model_container_uri = os.environ.get("MODEL_SERVING_CONTAINER_IMAGE_URI")
            managed_model_display_name = os.environ.get("MODEL_DISPLAY_NAME")
            if not managed_model_container_uri or not managed_model_display_name:
                raise ValueError("Managed model environment variables (MODEL_SERVING_CONTAINER_IMAGE_URI and MODEL_DISPLAY_NAME) must be set.")

        _cloud_trainer = GoogleCloudTrainer(
            project_id=project_id,
            location=location,
            service_account=service_account,
            enable_managed_model=enable_managed_model,
            managed_model_container_uri=managed_model_container_uri,
            managed_model_display_name=managed_model_display_name,
        )
    return _cloud_trainer
