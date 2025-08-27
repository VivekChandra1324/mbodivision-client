# app/main.py

# --- 1. Standard & Third-Party Imports ---
import asyncio
import base64
import io
import json
import logging
import os
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import ValidationError

# --- 2. Local Application Imports ---
from app.cloud_storage import CloudStorage, get_cloud_storage, test_cloud_storage_connection
from app.llm_verification import LLMVerifier
from app.models import BoundingBox, DataSubmissionResponse, LLMStatus
from app.training_pipeline import get_training_pipeline_manager
from app.utils import process_image
from app.yolo_service import YOLOService, get_yolo_service

# --- 3. Basic Configuration ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 4. FastAPI App Initialization ---
app = FastAPI(
    title="MbodiVision API",
    version="1.0.0",
    description="API for data submission, training, and inference for a YOLO object detection model."
)

# Configure templates for serving HTML files
templates = Jinja2Templates(directory="app/templates")

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 5. Application Startup & Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """
    Handles application startup logic:
    1. Tests cloud connections.
    2. Loads the best available model for inference.
    3. Starts the automated training pipeline monitor.
    """
    logger.info("🚀 Application starting up...")
    try:
        # Check that cloud services are reachable
        success, message = test_cloud_storage_connection()
        if not success:
            logger.error(f"❌ CRITICAL: Cloud storage connection failed: {message}")
            # In a real-world scenario, you might prevent the app from starting
        else:
            logger.info("✅ Cloud storage connection successful.")

        # Automatically load the best model from the cloud for the inference service
        logger.info("🔄 Checking for the best model to activate for inference...")
        await get_yolo_service().refresh_best_model()

        # Start the background pipeline that monitors for new data to train on
        if os.getenv("AUTO_START_PIPELINE", "true").lower() in ["1", "true", "yes"]:
            logger.info("🤖 Starting background training pipeline monitoring...")
            await get_training_pipeline_manager().start_monitoring()

    except Exception as e:
        logger.error(f"❌ An error occurred during application startup: {e}", exc_info=True)

# --- 6. HTML Serving Endpoints ---
@app.get("/", summary="Serve Data Upload Page")
async def serve_upload_page(request: Request):
    """Serves the main data submission page (`upload.html`)."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/inference", summary="Serve Inference Page")
async def serve_inference_page(request: Request):
    """Serves the object detection (inference) page (`inference.html`)."""
    return templates.TemplateResponse("inference.html", {"request": request})

# --- 7. Core API Endpoints ---
@app.post("/api/upload", response_model=DataSubmissionResponse, summary="Upload Image and Annotations")
async def handle_data_upload(
    image: UploadFile = File(..., description="The image file to be uploaded."),
    bounding_boxes_json: str = Form(..., description="A JSON string of bounding box annotations in pixel coordinates."),
    conversation: Optional[str] = Form(None, description="Optional text context about the image."),
    storage: CloudStorage = Depends(get_cloud_storage),
):
    """
    Receives an image and its metadata, saves it to persistent storage,
    and initiates a background verification task.
    """
    try:
        # Process image to get bytes, numpy array, and dimensions
        image_result = await process_image(image)
        image_bytes = image_result["content_bytes"]
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size

        # Parse and normalize bounding boxes from pixel to relative coordinates
        pixel_boxes = json.loads(bounding_boxes_json)
        if not pixel_boxes:
             raise HTTPException(status_code=400, detail="At least one bounding box is required.")

        normalized_boxes = [
            BoundingBox(
                x1=box["xmin"] / img_width,
                y1=box["ymin"] / img_height,
                x2=box["xmax"] / img_width,
                y2=box["ymax"] / img_height,
                label=box["name"],
            )
            for box in pixel_boxes
        ]

        # Save data to cloud storage with a PENDING status
        submission_id = await storage.save_data_async(
            image_array=image_result["numpy_array"],
            bounding_boxes=normalized_boxes,
            conversation=conversation,
            llm_status=LLMStatus.PENDING,
            original_image_bytes=image_bytes,
        )

        # Start the LLM verification as a non-blocking background task
        asyncio.create_task(
            verify_submission_background(
                submission_id=submission_id,
                image_bytes=image_bytes,
                bounding_boxes=normalized_boxes,
                conversation=conversation,
            )
        )

        logger.info(f"Successfully received submission {submission_id}. Verification started.")
        return DataSubmissionResponse(
            submission_id=submission_id,
            message="Data received successfully. LLM verification is in progress.",
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Upload failed due to invalid JSON format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid format for bounding_boxes_json: {e}")
    except ValidationError as e:
        logger.warning(f"Upload failed due to validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Bounding box data is invalid: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during the upload process.")


@app.post("/api/yolo/detect", summary="Perform Object Detection")
async def detect_objects_in_image(
    image: UploadFile = File(..., description="The image file to perform detection on."),
    conf: float = Form(0.3, description="The confidence threshold for detections."),
    yolo_service: YOLOService = Depends(get_yolo_service),
):
    """
    Performs object detection on an uploaded image using the best available model.
    """
    try:
        image_bytes = await image.read()
        result = await yolo_service.detect_objects(image_bytes, conf)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Detection failed due to an unknown error."))

        # The service returns the annotated image bytes directly
        annotated_base64 = result["annotated_image"]

        return {
            "detections": result.get("detections", []),
            "annotated_image": annotated_base64,
            "model_name": result.get("model_name"),
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred during detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during detection.")


# --- 8. Background Tasks ---
async def verify_submission_background(
    submission_id: str,
    image_bytes: bytes,
    bounding_boxes: List[BoundingBox],
    conversation: Optional[str],
):
    """
    A non-blocking background task that uses an LLM to verify a submission
    and notifies the training pipeline if the data is good.
    """
    storage = get_cloud_storage()
    try:
        logger.info(f"Starting LLM verification for submission: {submission_id}")
        verifier = LLMVerifier()
        result = await verifier.verify_labels(
            image_bytes=image_bytes,
            conversation=conversation,
            bounding_boxes=bounding_boxes,
            submission_id=submission_id,
        )
        llm_status = result.get("llm_status", LLMStatus.REJECTED)
        await storage.update_llm_status_async(submission_id, llm_status)
        logger.info(f"Verification for {submission_id} complete. Status: {llm_status.value}")

        # If verified, notify the training pipeline that new data is available
        if llm_status == LLMStatus.VERIFIED:
            pipeline_manager = get_training_pipeline_manager()
            all_verified_data = await storage.list_data_async()
            await pipeline_manager.notify_new_image_added(all_verified_data)

    except Exception as e:
        logger.error(f"Error during background verification for {submission_id}: {e}", exc_info=True)
        # Ensure submission is marked as failed if an error occurs
        await storage.update_llm_status_async(submission_id, LLMStatus.REJECTED)


# --- 9. Main Execution Block ---
if __name__ == "__main__":
    """
    This block allows running the application directly for development.
    In production, a process manager like Gunicorn with Uvicorn workers is recommended.
    Example: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
    """
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        env_file=".env"
    )