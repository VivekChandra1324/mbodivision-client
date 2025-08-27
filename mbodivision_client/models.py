"""
Client-side models for the MbodiVision client package.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """
    Represents a single bounding box with normalized coordinates.
    
    Coordinates (x1, y1) represent the top-left corner, and (x2, y2)
    represent the bottom-right corner.
    """
    x1: float = Field(..., ge=0.0, le=1.0, description="Left x coordinate (normalized 0-1)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top y coordinate (normalized 0-1)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right x coordinate (normalized 0-1)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom y coordinate (normalized 0-1)")
    label: str = Field(..., description="Class name/label for the object")

    @model_validator(mode='after')
    def check_coordinates_logic(self) -> 'BoundingBox':
        """
        Ensures that the coordinates are logical (e.g., x2 > x1).
        This validator runs *after* all individual fields have been validated.
        """
        if self.x1 >= self.x2:
            raise ValueError("x2 must be greater than x1")
        if self.y1 >= self.y2:
            raise ValueError("y2 must be greater than y1")
        return self

    @property
    def width(self) -> float:
        """Calculates the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Calculates the height of the bounding box."""
        return self.y2 - self.y1


class DataSubmissionResponse(BaseModel):
    """Defines the successful response structure for the /api/upload endpoint."""
    submission_id: str
    message: str


class DetectionResult(BaseModel):
    """Represents the result of an object detection operation."""
    detections: List[dict]
    annotated_image: str
    model_name: Optional[str] = None
