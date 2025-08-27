"""
MbodiVision App Package
"""

# Import key components for easy access
from app.models import LLMStatus, BoundingBox
from app.training_pipeline import TrainingPipelineManager, get_training_pipeline_manager
from app.yolo_service import YOLOService, get_yolo_service
from app.cloud_storage import CloudStorage, get_cloud_storage

__version__ = "1.0.0"
__all__ = [
    "LLMStatus",
    "BoundingBox", 
    "TrainingPipelineManager",
    "get_training_pipeline_manager",
    "YOLOService",
    "get_yolo_service",
    "CloudStorage",
    "get_cloud_storage"
]
