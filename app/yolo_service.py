"""
YOLO service module for managing cloud-model-only YOLO inference using Google Cloud Storage.
Automatically refreshes and activates the best available cloud model from pipeline state.
Downloads the active cloud model from GCS via CloudStorage, caches locally, and runs inference.
"""

import json
import logging
import io
import os
import tempfile
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ultralytics import YOLO
from PIL import Image

from app.cloud_storage import get_cloud_storage

logger = logging.getLogger(__name__)


class YOLOService:
    """Service for YOLO model cloud inference using GCS with auto-refresh of best model."""

    def __init__(self):
        self.active_model_path = None  # Cloud URI (gs://bucket/path/to/model.pt)
        self._model = None
        self.model_loaded = False
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self._local_model_path = None  # path to locally cached model file
        self.cloud_storage = get_cloud_storage()
        logger.info("Initialized YOLOService for cloud model inference.")

    async def refresh_best_model(self):
        """
        Query cloud storage pipeline state for the best model path,
        and activate it if different from the current active model.
        """
        try:
            pipeline_state = await self.cloud_storage.load_pipeline_state_async()
            if pipeline_state and pipeline_state.get("best_model_path"):
                best_path = pipeline_state["best_model_path"]
                if best_path != self.active_model_path:
                    success = await self.activate_model(best_path)
                    if success:
                        logger.info(f"Activated new best cloud model: {best_path}")
                    else:
                        logger.error(f"Failed to activate best cloud model: {best_path}")
            else:
                logger.warning("No best model path found in pipeline state.")
        except Exception as e:
            logger.error(f"Error refreshing best model from pipeline state: {e}", exc_info=True)

    async def activate_model(self, model_path: str) -> bool:
        """
        Activate a cloud model by GCS URI (must start with 'gs://').
        Deactivates any previously loaded model and clears cache.
        """
        if not model_path.startswith("gs://"):
            logger.error(f"Only cloud models with gs:// URI can be activated. Provided: {model_path}")
            return False

        self.active_model_path = model_path
        self._model = None
        self.model_loaded = False

        # Remove any existing local cached model before new download
        if self._local_model_path and os.path.exists(self._local_model_path):
            try:
                os.remove(self._local_model_path)
                logger.info(f"Removed old cached model file at {self._local_model_path}")
            except Exception as e:
                logger.warning(f"Could not remove old cached model file at {self._local_model_path}: {e}")

        self._local_model_path = None

        logger.info(f"Activated cloud model: {self.active_model_path}")
        return True

    async def detect_objects(self, image_bytes: bytes, conf: float = 0.3) -> dict:
        """Download cloud model if needed, run inference on image bytes, return detection results."""

        try:
            if not self.active_model_path or not self.active_model_path.startswith("gs://"):
                error_msg = "No cloud model activated for inference."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if not self.model_loaded:
                # Download the cloud model to local temp file
                logger.info(f"Downloading cloud model from {self.active_model_path} ...")
                model_bytes = await self.cloud_storage.download_file_async(self.active_model_path)
                if not model_bytes:
                    error_msg = f"Failed to download cloud model from {self.active_model_path}."
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}

                # Save to a temporary local file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
                    temp_file.write(model_bytes)
                    self._local_model_path = temp_file.name
                logger.info(f"Saved cloud model locally at {self._local_model_path}")

                # Load YOLO model
                self._model = YOLO(self._local_model_path)
                self.model_loaded = True
                logger.info("YOLO cloud model loaded successfully.")

            # Convert raw bytes to PIL Image then to NumPy array
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(pil_img)

            # Perform prediction on NumPy array image
            results = self._model.predict(source=img_np, conf=conf, save=False)
            result = results[0]

            # Create annotated image
            annotated_array = result.plot()
            annotated_pil = Image.fromarray(annotated_array)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="JPEG")
            annotated_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            img_width, img_height = annotated_pil.size

            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append({
                    "class_name": self._model.names[class_id],
                    "confidence": conf_score,
                    "x1": x1 / img_width,
                    "y1": y1 / img_height,
                    "x2": x2 / img_width,
                    "y2": y2 / img_height,
                })

            return {
                "success": True,
                "detections": detections,
                "annotated_image": annotated_base64,
                "model_name": self.active_model_path,
            }

        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_base_model_metrics(self) -> dict:
        # Not applicable since only cloud models are used
        return {}

    async def train_model_on_cloud(self, **training_params) -> dict:
        """
        Forwards training job submission to cloud trainer.
        """
        try:
            cloud_trainer = getattr(self.cloud_storage, 'get_cloud_trainer', None)
            if callable(cloud_trainer):
                cloud_trainer = cloud_trainer()
            else:
                from app.cloud_training import get_cloud_trainer  # fallback import
                cloud_trainer = get_cloud_trainer()

            dataset_yaml = training_params.get("dataset_yaml")
            if not dataset_yaml:
                raise ValueError("dataset_yaml parameter is required")

            cloud_training_args = {
                "dataset_yaml": dataset_yaml,
                "advanced_params": training_params.get("advanced_params", {}),
            }

            logger.info("Submitting cloud training job with params: %s", cloud_training_args)
            result = await cloud_trainer.submit_training_job(**cloud_training_args)

            if result.get("success", False):
                logger.info("Cloud training job submitted successfully.")
                job_info = result.get("job_info", {})
                self._save_training_job_info(job_info)
                return {"success": True, "message": "Cloud training job submitted", "job_info": job_info}
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Cloud training submission failed: {error}")
                return {"success": False, "error": error}

        except Exception as e:
            logger.error(f"Exception while submitting cloud training job: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _save_training_job_info(self, job_info: dict):
        try:
            model_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "models"
            jobs_file = model_dir / "cloud_jobs.json"
            jobs_file.parent.mkdir(parents=True, exist_ok=True)

            if jobs_file.exists():
                with open(jobs_file, "r") as f:
                    try:
                        jobs_data = json.load(f)
                    except json.JSONDecodeError:
                        jobs_data = {"jobs": []}
            else:
                jobs_data = {"jobs": []}

            jobs_data["jobs"].append(job_info)

            with open(jobs_file, "w") as f:
                json.dump(jobs_data, f, indent=2)

            logger.info(f"Saved cloud training job info with ID: {job_info.get('job_id', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed saving training job info: {e}")

# Singleton factory
_yolo_service = None


def get_yolo_service() -> YOLOService:
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YOLOService()
    return _yolo_service
