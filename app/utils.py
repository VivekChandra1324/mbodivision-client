import numpy as np
from PIL import Image
import io
from typing import Optional, Dict, Any, Tuple
import logging

from app.models import BoundingBox

# Setup logging
logger = logging.getLogger(__name__)

def validate_bounding_box(x1: float, y1: float, x2: float, y2: float, 
                          img_width: Optional[int] = None, img_height: Optional[int] = None) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        img_width, img_height: Optional image dimensions for additional validation
        
    Returns:
        True if valid, False otherwise
    """
    # Basic validation
    if not (x1 < x2 and y1 < y2):
        return False
    
    # If image dimensions are provided, check if within bounds
    if img_width is not None and img_height is not None:
        if not (0 <= x1 < img_width and 0 <= x2 <= img_width and
                0 <= y1 < img_height and 0 <= y2 <= img_height):
            return False
    
    return True

async def process_image(image_file):
    """
    Process an image file following the standard pipeline:
    1. Receive image file (UploadFile)
    2. Read bytes
    3. Normalize format (jpg / png)
    4. Convert to NumPy array → for internal processing
    
    Args:
        image_file: FastAPI UploadFile object
        
    Returns:
        Dictionary with processed image data:
        {
            'numpy_array': np.ndarray,  # NumPy array for processing and storage
            'format': str,  # Normalized format ('jpg' or 'png')
            'content_bytes': bytes,  # Original file bytes
        }
        
    Raises:
        ValueError: If the image file is invalid or cannot be processed
    """
    try:
        # Read image bytes
        content = await image_file.read()
        
        if not content:
            raise ValueError("Empty image file")
        
        # Extract and normalize format
        image_format = image_file.filename.split('.')[-1].lower()
        if image_format in ['jpg', 'jpeg']:
            normalized_format = 'jpg'
        elif image_format == 'png':
            normalized_format = 'png'
        else:
            normalized_format = 'png'  # Default to PNG for unknown formats
            
        # Try to open the image to validate it
        try:
            img = Image.open(io.BytesIO(content))
            # Convert to NumPy array
            numpy_array = np.array(img)
        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}")
        
        result = {
            'format': normalized_format,
            'content_bytes': content,
            'numpy_array': numpy_array
        }
            
        return result
        
    except Exception as e:
        # Re-raise as ValueError with descriptive message
        raise ValueError(f"Failed to process image file: {str(e)}")
