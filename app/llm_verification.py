"""
Module for LLM verification of image labels and data quality using Google Vertex AI.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io

from app.models import LLMStatus, BoundingBox
from app.vertex_ai import VertexAI

logger = logging.getLogger(__name__)

class LLMVerifier:
    """Class to handle verification of labels and data quality using LLMs."""
    
    def __init__(self, model_provider: str = "vertex_ai"):
        """
        Initialize LLM verifier.
        
        Args:
            model_provider: The LLM provider to use (currently only supports vertex_ai)
        """
        self.model_provider = model_provider
        logger.info(f"Initialized LLM verifier with {model_provider}")
    
    async def verify_labels(self, 
                          image_bytes: bytes,
                          conversation: Optional[str] = None,
                          bounding_boxes: Optional[List[BoundingBox]] = None,
                          submission_id: str = None) -> Dict:
        """
        Verify the quality of labels using LLM.
        
        Args:
            image_bytes: The original image bytes
            conversation: The conversation text associated with the image
            bounding_boxes: List of bounding boxes
            submission_id: The ID of the submission
            
        Returns:
            Dictionary with verification results (verified or rejected)
        """
        try:
            logger.info(f"Verifying submission {submission_id} with {self.model_provider}")
            
            if self.model_provider == "vertex_ai":
                # Get the Vertex AI client
                vertex_ai = VertexAI()
                
                # Convert bounding boxes to dict format for Vertex AI
                bbox_dicts = []
                if bounding_boxes:
                    for bbox in bounding_boxes:
                        bbox_dict = {
                            "x1": bbox.x1,
                            "y1": bbox.y1,
                            "x2": bbox.x2,
                            "y2": bbox.y2,
                            "label": bbox.label or "unlabeled"
                        }
                        bbox_dicts.append(bbox_dict)
                
                # Verify the image using Vertex AI
                status, explanation = await vertex_ai.verify_image_async(
                    image_bytes=image_bytes,
                    conversation_text=conversation,
                    bounding_boxes=bbox_dicts
                )
                
                # Simplify to only verified or rejected
                if status.value in ["verified", "rejected"]:
                    verification_status = status.value
                    llm_status = status
                else:
                    # Convert any other status to rejected
                    verification_status = "rejected"
                    llm_status = LLMStatus.REJECTED
                    explanation = f"Status '{status.value}' converted to rejected for simplicity"
                
                logger.info(f"Verification result for {submission_id}: {verification_status}")
                
                return {
                    "submission_id": submission_id,
                    "verification_status": verification_status,
                    "message": explanation,
                    "llm_status": llm_status
                }
            else:
                # Fallback to auto-verification for unsupported providers
                logger.warning(f"Unsupported model provider: {self.model_provider}, auto-verifying")
                return {
                    "submission_id": submission_id,
                    "verification_status": "verified",
                    "message": f"Auto-verified (unsupported model provider: {self.model_provider})",
                    "llm_status": LLMStatus.VERIFIED
                }
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            # Convert errors to rejected status
            return {
                "submission_id": submission_id,
                "verification_status": "rejected",
                "message": f"Verification failed: {str(e)}",
                "llm_status": LLMStatus.REJECTED
            }
