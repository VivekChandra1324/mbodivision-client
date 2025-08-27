"""
Vertex AI Pipeline for automated YOLO training with enhanced robustness and error handling.

This script orchestrates continuous training by monitoring new data,
batching, job submission, monitoring, evaluation, and model promotion.
State is persisted for resilience and continuity.
"""

import asyncio
import logging
import os
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from app.cloud_storage import get_cloud_storage
from app.yolo_service import get_yolo_service
from app.cloud_training import get_cloud_trainer
from app.dataset_converter import YOLODatasetConverter

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Suppress verbose INFO logs from the cloud_storage module
logging.getLogger("app.cloud_storage").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- Data Classes ---
@dataclass
class TrainingBatch:
    """Represents a single training batch and its state."""
    batch_id: str
    image_count: int
    timestamp: datetime
    submissions: List
    status: str = "queued"
    training_job_id: Optional[str] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None
    output_dir: Optional[str] = None


# --- Core Pipeline Manager ---
class TrainingPipelineManager:
    """Manages automated, event-driven, persistent YOLO training lifecycle."""

    def __init__(self):
        self.cloud_storage = get_cloud_storage()
        self.yolo_service = get_yolo_service()
        self.threshold = int(os.getenv("TRAINING_THRESHOLD", "5"))

        # Persistent state
        self.total_images_processed = 0
        self.images_in_queue = 0
        self.training_in_progress = False
        self.batch_counter = 0

        # Async control
        self._monitoring_tasks: List[asyncio.Task] = []
        self.new_image_event = asyncio.Event()
        self.training_queue = asyncio.Queue()

        # Best model tracking
        self.best_model_path: Optional[str] = None
        self.best_model_metrics: Optional[Dict] = None

        logger.info(f"Initialized TrainingPipelineManager with threshold {self.threshold}. Loading state...")

    def _get_current_state(self) -> Dict:
        """Consolidates all persistent state variables into a single dictionary."""
        return {
            "total_images_processed": self.total_images_processed,
            "batch_counter": self.batch_counter,
            "best_model_path": self.best_model_path,
            "best_model_metrics": self.best_model_metrics,
        }

    async def initialize(self):
        """Load pipeline state from persistent storage."""
        loop = asyncio.get_event_loop()
        loaded_state = await loop.run_in_executor(None, self.cloud_storage.load_pipeline_state)

        if loaded_state:
            self.total_images_processed = loaded_state.get("total_images_processed", 0)
            self.batch_counter = loaded_state.get("batch_counter", 0)
            self.best_model_path = loaded_state.get("best_model_path")
            self.best_model_metrics = loaded_state.get("best_model_metrics")
            logger.info("✅ Pipeline state loaded.")
        else:
            logger.info("ℹ️ No saved pipeline state found. Starting fresh.")

    async def notify_new_image_added(self, submissions: List[Dict]):
        """Notify pipeline of new verified images and trigger batch creation if threshold met."""
        try:
            accounted_for = self.total_images_processed + self.images_in_queue
            new_image_count = max(0, len(submissions) - accounted_for)

            if new_image_count > 0:
                needed = self.threshold - new_image_count
                if needed > 0:
                    logger.info(f"✅ New images: {new_image_count}/{self.threshold}. Need {needed} more.")
                else:
                    logger.info(f"✅ Threshold met: {new_image_count} images available for training.")
            else:
                logger.info("ℹ️ No new verified images waiting.")

            if new_image_count >= self.threshold:
                self.new_image_event.set()
        except Exception as e:
            logger.error(f"Error in new image notification: {e}", exc_info=True)

    async def start_monitoring(self):
        """Begin asynchronous image processing and training execution loops."""
        if self._monitoring_tasks:
            return

        await self.initialize()
        logger.info("🚀 Starting training pipeline monitoring...")

        self._monitoring_tasks.append(asyncio.create_task(self._image_processor_loop()))
        self._monitoring_tasks.append(asyncio.create_task(self._training_executor_loop()))

        all_submissions = await self.cloud_storage.list_data_async()
        await self.notify_new_image_added(all_submissions)

    async def stop_monitoring(self):
        """Stop monitoring tasks gracefully."""
        for task in self._monitoring_tasks:
            task.cancel()
        self._monitoring_tasks.clear()
        logger.info("🛑 Training pipeline monitoring stopped.")

    async def _image_processor_loop(self):
        """Create training batches when threshold is met."""
        while True:
            try:
                await self.new_image_event.wait()
                self.new_image_event.clear()

                submissions = await self.cloud_storage.list_data_async()
                accounted_for = self.total_images_processed + self.images_in_queue
                new_image_count = max(0, len(submissions) - accounted_for)

                if new_image_count >= self.threshold:
                    self.batch_counter += 1
                    # The batch_id is still used internally for unique dataset naming
                    batch_id = f"batch_{self.batch_counter:03d}"
                    batch_submissions = submissions[accounted_for : accounted_for + new_image_count]

                    batch = TrainingBatch(
                        batch_id=batch_id,
                        image_count=len(batch_submissions),
                        timestamp=datetime.utcnow(),
                        submissions=batch_submissions,
                    )

                    await self.training_queue.put(batch)
                    self.images_in_queue += batch.image_count

            except asyncio.CancelledError:
                logger.info("Image processor loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in image processor loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _training_executor_loop(self):
        """Run training lifecycle for each batch sequentially."""
        while True:
            try:
                batch = await self.training_queue.get()

                self.training_in_progress = True
                logger.info("🚀 Starting training for new batch...")
                await self._run_and_monitor_batch(batch)

                self.images_in_queue = max(0, self.images_in_queue - batch.image_count)

                if batch.status == "completed":
                    self.total_images_processed += batch.image_count
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        self.cloud_storage.save_pipeline_state,
                        self._get_current_state(),
                    )

                self.training_in_progress = False
                self.training_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Training executor loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Critical error in training executor loop: {e}", exc_info=True)
                self.training_in_progress = False

    async def _run_and_monitor_batch(self, batch: TrainingBatch):
        """Submit, monitor, evaluate training job, and promote model if better."""
        try:
            dataset_name = f"{batch.batch_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            dataset_yaml, _ = await self._prepare_dataset(dataset_name, batch.submissions)
            if not dataset_yaml:
                raise RuntimeError("Failed to prepare dataset.")

            cloud_trainer = get_cloud_trainer()
            advanced_params = {"patience": 10}

            if self.best_model_path:
                advanced_params["model"] = self.best_model_path
                logger.info(f"🔄 Fine-tuning with best model: {self.best_model_path}")
            else:
                logger.info("ℹ️ Training from scratch (no existing best model).")

            submission_result = await cloud_trainer.submit_training_job(
                dataset_yaml=dataset_yaml, advanced_params=advanced_params
            )
            if not submission_result.get("success"):
                raise RuntimeError(f"Job submission failed: {submission_result.get('error')}")

            job_info = submission_result["job_info"]
            batch.training_job_id = job_info["job_id"]
            batch.status = "training"
            logger.info(f"✅ Training job submitted. Job ID: {job_info['job_id']}")

            monitor_result = await cloud_trainer.monitor_training_job(batch.training_job_id)
            if not monitor_result.get("success") or monitor_result.get("status") != "COMPLETED":
                raise RuntimeError(f"Training job failed or was cancelled: {monitor_result.get('error')}")

            batch.output_dir = monitor_result.get("output_dir")
            logger.info("🎉 Training job completed.")

            results_paths = monitor_result.get("results", {})
            model_path = results_paths.get("best_model_path")
            if not model_path:
                raise RuntimeError("Missing best_model_path in training results.")

            training_metrics = await self._evaluate_batch_results(results_paths)

            logger.info("--- Validation Metrics ---")
            for key, value in training_metrics.items():
                logger.info(f"{key}: {value}")
            logger.info("------------------------")

            is_first_model = not self.best_model_metrics
            is_better = False
            if not is_first_model:
                new_metric = training_metrics.get("mAP50-95", 0.0)
                current_metric = self.best_model_metrics.get("mAP50-95", 0.0)
                is_better = new_metric > current_metric

            if is_first_model or is_better:
                if is_first_model:
                    logger.info(f"🏆 Initial best model trained with mAP50-95: {training_metrics.get('mAP50-95', 0.0)}")
                else:
                    logger.info(f"🏆 New best model found with mAP50-95: {training_metrics.get('mAP50-95', 0.0)}")

                self.best_model_path = model_path
                self.best_model_metrics = training_metrics
                await self.yolo_service.activate_model(model_path)

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self.cloud_storage.save_pipeline_state,
                    self._get_current_state(),
                )
            else:
                logger.info(
                    f"New model (mAP50-95: {training_metrics.get('mAP50-95', 0.0)}) "
                    f"did not outperform current best (mAP50-95: {self.best_model_metrics.get('mAP50-95', 0.0)})."
                )
            batch.status = "completed"

        except Exception as e:
            logger.error(f"Error during training run: {e}", exc_info=True)
            batch.status = "failed"

    async def _prepare_dataset(self, dataset_name: str, submissions: List) -> Tuple[Optional[str], Dict]:
        """Prepare YOLO dataset in GCS asynchronously."""
        try:
            converter = YOLODatasetConverter(self.cloud_storage)
            return await converter.prepare_dataset(dataset_name, submissions)
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}", exc_info=True)
            return None, {}

    async def _evaluate_batch_results(self, results_paths: Dict) -> Dict[str, float]:
        """Download and parse results.csv, extracting key validation metrics."""
        try:
            results_csv_path = results_paths.get("results_csv_path")
            if not results_csv_path:
                raise FileNotFoundError("results.csv path not found.")

            with tempfile.TemporaryDirectory() as temp_dir:
                local_csv_path = Path(temp_dir) / "results.csv"
                content = await self.cloud_storage.download_file_async(results_csv_path)
                if not content:
                    raise FileNotFoundError(f"Failed to download results.csv from {results_csv_path}")

                local_csv_path.write_bytes(content)

                df = pd.read_csv(local_csv_path)
                df.columns = df.columns.str.strip()
                last_row = df.iloc[-1]

                metric_map = {
                    "mAP50-95": ["metrics/mAP50-95(B)"],
                    "mAP50": ["metrics/mAP50(B)"],
                    "Precision": ["metrics/precision(B)"],
                    "Recall": ["metrics/recall(B)"],
                }

                extracted_metrics = {}
                for metric_name, possible_cols in metric_map.items():
                    for col in possible_cols:
                        if col in last_row:
                            # Explicitly cast to a standard Python float to prevent serialization errors.
                            value = float(last_row[col])
                            extracted_metrics[metric_name] = round(value, 4)
                            break
                    if metric_name not in extracted_metrics:
                        extracted_metrics[metric_name] = 0.0
                
                return extracted_metrics

        except Exception as e:
            logger.error(f"Failed to evaluate batch results: {e}", exc_info=True)
            return {"mAP50-95": 0.0, "mAP50": 0.0, "Precision": 0.0, "Recall": 0.0}


# --- Singleton Factory ---
_training_pipeline_manager: Optional[TrainingPipelineManager] = None

def get_training_pipeline_manager() -> TrainingPipelineManager:
    """Returns a thread-safe singleton instance of TrainingPipelineManager."""
    global _training_pipeline_manager
    if _training_pipeline_manager is None:
        _training_pipeline_manager = TrainingPipelineManager()
    return _training_pipeline_manager