"""
MbodiVision Client Package

A Python client for interacting with the MbodiVision API.
"""

from .client import MbodiVisionClient
from .models import BoundingBox, DataSubmissionResponse, DetectionResult

__version__ = "1.0.0"
__all__ = ["MbodiVisionClient", "BoundingBox", "DataSubmissionResponse", "DetectionResult"]
