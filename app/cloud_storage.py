"""
Cloud storage implementation using Google Cloud Storage and PostgreSQL.

This module provides a unified service class, `CloudStorage`, to handle all
interactions with cloud infrastructure. It is responsible for:
- Uploading image data (NumPy arrays, original files, in-memory bytes) to GCS.
- Storing submission metadata (paths, bounding boxes, status) in PostgreSQL.
- Storing and retrieving the training pipeline's persistent state.
- Providing methods to list, download, and update records.
- Offering asynchronous wrappers for all I/O-bound operations to ensure
  non-blocking performance in async applications.
"""

# 1. Standard Library Imports
import asyncio
import functools
import io
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# 2. Third-party Library Imports
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage as gcs
from google.cloud.exceptions import GoogleCloudError
from PIL import Image
from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

# 3. Local Application Imports
# Assuming models are defined in a sibling module or a structured app folder
from app.models import BoundingBox, LLMStatus, SubmissionRecord

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()
Base = declarative_base()


# --- Database Models ---
class SubmissionDB(Base):
    """Database ORM model for the 'submissions' table."""
    __tablename__ = 'submissions'
    submission_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    gcs_numpy_path = Column(String, nullable=False)
    gcs_original_path = Column(String, nullable=True)
    bounding_boxes = Column(JSON, nullable=False, default=[])
    conversation = Column(Text, nullable=True)
    llm_status = Column(String, nullable=False, default=LLMStatus.PENDING.value)

class PipelineStateDB(Base):
    """Database ORM model for storing the pipeline's persistent state."""
    __tablename__ = 'pipeline_state'
    state_key = Column(String, primary_key=True, default='singleton_state')
    total_images_processed = Column(Integer, nullable=False, default=0)
    best_model_path = Column(String, nullable=True)
    best_model_metrics = Column(JSON, nullable=True)
    batch_counter = Column(Integer, nullable=False, default=0) # Added for persistence


# --- Core Storage Service ---
class CloudStorage:
    """Manages all interactions with Google Cloud Storage and PostgreSQL."""

    def __init__(self, gcs_bucket_name: str, postgres_url: str, max_retries: int = 3, retry_delay: int = 1):
        self.gcs_bucket_name = gcs_bucket_name
        self.postgres_url = postgres_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self._init_gcs_client()
        self._init_postgres_connection()

    def _init_gcs_client(self):
        """Initializes the Google Cloud Storage client with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.gcs_client = gcs.Client()
                self.bucket = self.gcs_client.bucket(self.gcs_bucket_name)
                if not self.bucket.exists():
                    raise FileNotFoundError(f"GCS bucket '{self.gcs_bucket_name}' not found or inaccessible.")
                logger.info(f"✅ GCS bucket connected: {self.gcs_bucket_name}")
                return
            except Exception as e:
                logger.warning(f"GCS init attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"GCS initialization failed after {self.max_retries} attempts.") from e

    def _init_postgres_connection(self):
        """Initializes the PostgreSQL database connection and sessionmaker with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.engine = create_engine(self.postgres_url, pool_pre_ping=True)
                with self.engine.connect(): # Test connection
                    Base.metadata.create_all(bind=self.engine)
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                logger.info("✅ PostgreSQL connected and tables verified.")
                return
            except SQLAlchemyError as e:
                logger.warning(f"PostgreSQL init attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"PostgreSQL connection failed after {self.max_retries} attempts.") from e

    # --- Synchronous Core Methods ---

    def save_data(
        self,
        image_array: np.ndarray,
        bounding_boxes: List[BoundingBox],
        conversation: Optional[str] = None,
        llm_status: LLMStatus = LLMStatus.PENDING,
        original_image_bytes: Optional[bytes] = None,
    ) -> str:
        """Saves image data to GCS and metadata to PostgreSQL."""
        submission_id = str(uuid.uuid4())
        gcs_paths = {}
        try:
            gcs_paths = self._upload_to_gcs(
                submission_id=submission_id,
                image_array=image_array,
                original_image_bytes=original_image_bytes
            )
            record = SubmissionRecord(
                submission_id=submission_id,
                timestamp=datetime.utcnow(),
                gcs_numpy_path=gcs_paths["numpy_path"],
                gcs_original_path=gcs_paths.get("original_path"),
                bounding_boxes=bounding_boxes,
                conversation=conversation,
                llm_status=llm_status,
            )
            self._save_metadata_to_postgres(record)
            logger.info(f"Submission saved successfully: {submission_id}")
            return submission_id
        except Exception as e:
            logger.error(f"Failed to save submission {submission_id}: {e}", exc_info=True)
            if gcs_paths:
                self._cleanup_failed_upload(gcs_paths)
            raise RuntimeError(f"Failed to save submission: {e}") from e

    def list_data(self) -> List[Dict]:
        """Lists all VERIFIED submissions from the database."""
        with self.SessionLocal() as db_session:
            verified_records = (
                db_session.query(SubmissionDB)
                .filter(SubmissionDB.llm_status == LLMStatus.VERIFIED.value)
                .order_by(SubmissionDB.timestamp)
                .all()
            )
            return [
                {
                    'image_path': record.gcs_original_path,
                    'submission_id': str(record.submission_id),
                    'bounding_boxes': record.bounding_boxes,
                }
                for record in verified_records if record.gcs_original_path
            ]

    def download_file(self, gcs_path: str) -> Optional[bytes]:
        """Downloads a file from GCS given its full gs:// path or relative path."""
        try:
            if gcs_path.startswith("gs://"):
                parsed_uri = urlparse(gcs_path)
                bucket_name = parsed_uri.netloc
                blob_name = parsed_uri.path.lstrip('/')
                target_bucket = self.gcs_client.bucket(bucket_name)
                blob = target_bucket.blob(blob_name)
            else:
                blob = self.bucket.blob(gcs_path)

            if not blob or not blob.exists():
                logger.error(f"File not found in GCS: {gcs_path}")
                return None
            
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error downloading file {gcs_path}: {e}", exc_info=True)
            return None

    def update_llm_status(self, submission_id: str, llm_status: LLMStatus):
        """Updates the LLM verification status for a specific submission."""
        with self.SessionLocal() as db_session:
            try:
                record = db_session.query(SubmissionDB).filter_by(submission_id=uuid.UUID(submission_id)).first()
                if record:
                    record.llm_status = llm_status.value
                    db_session.commit()
                    logger.info(f"LLM status for {submission_id} updated to {llm_status.value}")
                else:
                    logger.warning(f"Submission {submission_id} not found for status update.")
            except SQLAlchemyError as e:
                db_session.rollback()
                logger.error(f"Failed to update LLM status for {submission_id}: {e}")
                raise

    def save_pipeline_state(self, state: Dict):
        """Saves or updates the training pipeline's state in the database."""
        with self.SessionLocal() as db_session:
            try:
                state_record = db_session.query(PipelineStateDB).filter_by(state_key='singleton_state').first()
                if not state_record:
                    state_record = PipelineStateDB(state_key='singleton_state')
                    db_session.add(state_record)
                
                # Update fields from the state dictionary
                for key, value in state.items():
                    if hasattr(state_record, key):
                        setattr(state_record, key, value)
                
                db_session.commit()
                logger.info(f"💾 Pipeline state saved: {state}")
            except SQLAlchemyError as e:
                db_session.rollback()
                logger.error(f"Failed to save pipeline state: {e}")
                raise

    def load_pipeline_state(self) -> Optional[Dict]:
        """Loads the training pipeline's state from the database."""
        with self.SessionLocal() as db_session:
            state_record = db_session.query(PipelineStateDB).filter_by(state_key='singleton_state').first()
            if state_record:
                state = {
                    "total_images_processed": state_record.total_images_processed,
                    "best_model_path": state_record.best_model_path,
                    "best_model_metrics": state_record.best_model_metrics,
                    "batch_counter": state_record.batch_counter,
                }
                logger.info(f"✅ Pipeline state loaded from DB: Processed {state['total_images_processed']} images.")
                return state
            logger.info("ℹ️ No previous pipeline state found in DB. Starting fresh.")
            return None

    def upload_file_from_bytes(self, gcs_path: str, file_bytes: bytes, content_type: str):
        """Uploads a byte string to a specified path in GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(file_bytes, content_type=content_type, timeout=120)
        except GoogleCloudError as e:
            logger.error(f"Failed to upload file to {gcs_path}: {e}")
            raise

    # --- Asynchronous Wrappers ---

    async def save_data_async(self, *args, **kwargs) -> str:
        """Asynchronously saves image data and metadata."""
        loop = asyncio.get_event_loop()
        func_call = functools.partial(self.save_data, *args, **kwargs)
        return await loop.run_in_executor(self.thread_pool, func_call)

    async def list_data_async(self) -> List[Dict]:
        """Asynchronously lists all verified submissions."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self.list_data)

    async def download_file_async(self, gcs_path: str) -> Optional[bytes]:
        """Asynchronously downloads a file from GCS."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self.download_file, gcs_path)

    async def update_llm_status_async(self, submission_id: str, llm_status: LLMStatus):
        """Asynchronously updates the LLM status for a given submission."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, self.update_llm_status, submission_id, llm_status)

    async def upload_file_from_bytes_async(self, gcs_path: str, file_bytes: bytes, content_type: str):
        """Asynchronously uploads a byte string to GCS."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_pool, self.upload_file_from_bytes, gcs_path, file_bytes, content_type
        )

    async def load_pipeline_state_async(self) -> Optional[Dict]:
        """Asynchronously loads the pipeline's persistent state."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self.load_pipeline_state)

    # --- Private Helper Methods ---

    def _upload_to_gcs(
        self, submission_id: str, image_array: np.ndarray, original_image_bytes: Optional[bytes]
    ) -> Dict[str, str]:
        """Handles the logic of uploading numpy array and original image to GCS."""
        paths = {}
        # Upload NumPy array
        numpy_path = f"numpy_arrays/{submission_id}.npy"
        buffer = io.BytesIO()
        np.save(buffer, image_array)
        buffer.seek(0)
        self.bucket.blob(numpy_path).upload_from_file(buffer, content_type='application/octet-stream', timeout=120)
        paths["numpy_path"] = f"gs://{self.gcs_bucket_name}/{numpy_path}"

        # Upload original image if provided, ensuring it's JPEG
        if original_image_bytes:
            jpeg_bytes = self._ensure_jpeg_format(original_image_bytes)
            original_path = f"original_images/{submission_id}.jpg"
            self.bucket.blob(original_path).upload_from_string(jpeg_bytes, content_type='image/jpeg', timeout=120)
            paths["original_path"] = f"gs://{self.gcs_bucket_name}/{original_path}"
        
        return paths

    def _save_metadata_to_postgres(self, record: SubmissionRecord):
        """Handles the logic of saving submission metadata to PostgreSQL."""
        with self.SessionLocal() as db_session:
            try:
                db_record = SubmissionDB(
                    submission_id=record.submission_id,
                    timestamp=record.timestamp,
                    gcs_numpy_path=record.gcs_numpy_path,
                    gcs_original_path=record.gcs_original_path,
                    bounding_boxes=[bbox.model_dump() for bbox in record.bounding_boxes],
                    conversation=record.conversation,
                    llm_status=record.llm_status.value
                )
                db_session.add(db_record)
                db_session.commit()
            except SQLAlchemyError as e:
                db_session.rollback()
                logger.error(f"PostgreSQL save failed for {record.submission_id}: {e}", exc_info=True)
                raise

    def _cleanup_failed_upload(self, gcs_paths: Dict[str, str]):
        """Deletes files from GCS if the corresponding database transaction fails."""
        logger.warning(f"Cleaning up failed submission. Deleting GCS files: {gcs_paths}")
        for path in gcs_paths.values():
            if path:
                try:
                    blob_name = path.replace(f"gs://{self.gcs_bucket_name}/", "")
                    self.bucket.blob(blob_name).delete()
                    logger.info(f"Cleaned up GCS file: {path}")
                except Exception as e:
                    logger.error(f"Failed to clean up GCS file {path}: {e}")

    def _ensure_jpeg_format(self, image_bytes: bytes) -> bytes:
        """Converts image bytes to JPEG format if they are not already."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.format == 'JPEG':
                return image_bytes
            # Convert to RGB if it has an alpha channel (like PNG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='JPEG', quality=95)
            return output_buffer.getvalue()
        except Exception as e:
            logger.warning(f"Could not convert image to JPEG, returning original bytes. Error: {e}")
            return image_bytes

# --- Singleton Factory ---
_cloud_storage_instance: Optional[CloudStorage] = None

def get_cloud_storage() -> CloudStorage:
    """
    Returns a thread-safe singleton instance of the CloudStorage service.
    Initializes the instance on first call.
    """
    global _cloud_storage_instance
    if _cloud_storage_instance is None:
        gcs_bucket_name = os.getenv('GCS_BUCKET_NAME')
        postgres_url = os.getenv('POSTGRES_URL')
        
        if not all([gcs_bucket_name, postgres_url]):
            raise RuntimeError("Missing required environment variables: GCS_BUCKET_NAME or POSTGRES_URL.")
            
        _cloud_storage_instance = CloudStorage(
            gcs_bucket_name=gcs_bucket_name,
            postgres_url=postgres_url
        )
    return _cloud_storage_instance

def test_cloud_storage_connection() -> Tuple[bool, str]:
    """
    Tests the connection to both GCS and PostgreSQL.
    Returns a tuple of (success_boolean, message_string).
    """
    try:
        get_cloud_storage()
        return True, "Cloud storage connections successful."
    except Exception as e:
        logger.error(f"Cloud storage connection test failed: {e}", exc_info=True)
        return False, str(e)
