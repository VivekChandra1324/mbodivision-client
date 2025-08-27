"""
Client for interacting with the MbodiVision API.
"""

import httpx
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import io
import os
import json

from .models import BoundingBox, DataSubmissionResponse, DetectionResult

class MbodiVisionClient:
    """
    Client for interacting with the MbodiVision API.
    
    This client uses httpx.AsyncClient for all operations, which means:
    1. All methods must be awaited
    2. The client should be used within an async context manager
       using `async with MbodiVisionClient() as client:`
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client with the API base URL.
        
        Args:
            base_url: The base URL of the MbodiVision API
            
        Note:
            For proper resource management, use this client as an async context manager:
            ```
            async with MbodiVisionClient() as client:
                # Use client here
            ```
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)  # Increased timeout for larger images
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def submit_data_with_dict_bbox(self,
                           image_path: str,
                           bbox_dict: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                           conversation: Optional[str] = None) -> DataSubmissionResponse:
        """
        Submit data to the MbodiVision API with bounding boxes in dictionary format.
        
        Args:
            image_path: Path to the image file (MANDATORY)
            bbox_dict: Optional dictionary of bounding boxes in format {label: (x1, y1, x2, y2)}
            conversation: Optional conversation string
            
        Returns:
            API response as DataSubmissionResponse object
        """
        # Convert bounding box dictionary to list of BoundingBox objects if provided
        bounding_boxes = None
        if bbox_dict:
            bounding_boxes = self.create_bounding_boxes_from_dict(bbox_dict)
            
        # Call the submit_data method with file path
        return await self.submit_data(
            image_path=image_path,
            bounding_boxes=bounding_boxes,
            conversation=conversation
        )
        
    async def submit_data(self,
                    image_path: str,
                    bounding_boxes: Optional[List[BoundingBox]] = None,
                    bbox_dict: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                    conversation: Optional[str] = None) -> DataSubmissionResponse:
        """
        Submit data to the MbodiVision API using direct file upload.
        
        Args:
            image_path: Path to the image file (MANDATORY)
            bounding_boxes: Optional list of BoundingBox objects
            bbox_dict: Optional dictionary of bounding boxes in format {label: (x1, y1, x2, y2)}
            conversation: Optional conversation string
            
        Returns:
            API response as DataSubmissionResponse object
        """
        files = {"image": open(image_path, "rb")}
        
        # Prepare form data
        data = {}
        if conversation:
            data["conversation"] = conversation
            
        # Process bounding boxes - only one format should be provided
        if bounding_boxes and bbox_dict:
            raise ValueError("Cannot provide both 'bounding_boxes' and 'bbox_dict'. Please use only one format.")
        
        if bbox_dict:
            bounding_boxes = self.create_bounding_boxes_from_dict(bbox_dict)
            
        # Only add bounding_boxes_json if we actually have bounding boxes
        if bounding_boxes and len(bounding_boxes) > 0:
            # Convert bounding boxes to JSON
            boxes_data = []
            for box in bounding_boxes:
                # Validate that each bounding box has a label
                if not box.label:
                    raise ValueError("All bounding boxes must have labels")
                
                box_dict = {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "label": box.label  # Label is now required
                }
                boxes_data.append(box_dict)
            
            # Only add to data if we have actual boxes
            if boxes_data:
                data["bounding_boxes_json"] = json.dumps(boxes_data)
        
        # Send the request
        try:
            # Use the /api/upload endpoint which is designed for file uploads
            response = await self.client.post(
                f"{self.base_url}/api/upload",
                files=files,
                data=data
            )
            
            # Raise exception for bad responses
            response.raise_for_status()
            
            # Close file handles
            for file in files.values():
                file.close()
            
            # Return the response as a DataSubmissionResponse object
            response_data = response.json()
            return DataSubmissionResponse(**response_data)
        except Exception as e:
            # Make sure to close file handles even if there's an error
            for file in files.values():
                file.close()
            raise e

    async def detect_objects(self, image_path: str, conf: float = 0.3) -> DetectionResult:
        """
        Perform object detection on an image using the MbodiVision API.
        
        Args:
            image_path: Path to the image file
            conf: Confidence threshold for detections (default: 0.3)
            
        Returns:
            DetectionResult object containing detections and annotated image
        """
        files = {"image": open(image_path, "rb")}
        data = {"conf": str(conf)}
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/yolo/detect",
                files=files,
                data=data
            )
            
            response.raise_for_status()
            
            # Close file handles
            for file in files.values():
                file.close()
            
            # Return the response as a DetectionResult object
            response_data = response.json()
            return DetectionResult(**response_data)
        except Exception as e:
            # Make sure to close file handles even if there's an error
            for file in files.values():
                file.close()
            raise e

    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate that an image file exists and can be opened.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if the image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Just checking if the image can be opened
                return True
        except:
            return False
        
    @staticmethod
    def create_bounding_boxes_from_dict(bbox_dict: Dict[str, Tuple[float, float, float, float]]) -> List[BoundingBox]:
        """
        Create BoundingBox objects from a dictionary of the format {label: (x1, y1, x2, y2)}
        
        Args:
            bbox_dict: Dictionary mapping labels to coordinate tuples
            
        Returns:
            List of BoundingBox objects
        """
        bounding_boxes = []
        for label, coords in bbox_dict.items():
            if len(coords) != 4:
                raise ValueError(f"Coordinates must be a tuple of 4 values (x1, y1, x2, y2), got {coords}")
            x1, y1, x2, y2 = coords
            bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        return bounding_boxes
