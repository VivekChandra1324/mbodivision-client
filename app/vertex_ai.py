"""
Module for Google Vertex AI integration for LLM-based image verification.
"""

import os
import io
import logging
import json
import asyncio
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account

from app.models import LLMStatus, BoundingBox

logger = logging.getLogger(__name__)

class VertexAI:
    """Handles interaction with Google Vertex AI for LLM verification using Gemini models."""
    
    def __init__(self, 
                 credentials_path: Optional[str] = None,
                 project_id: Optional[str] = None, 
                 location: str = "us-central1",
                 model_name: str = "gemini-2.0-flash-001",  # Using gemini-2.0-flash-001 as requested
                 max_retries: int = 3,
                 retry_delay: int = 1):
        """
        Initialize Vertex AI client with Gemini model.
        
        Args:
            credentials_path: Path to service account key file
            project_id: GCP project ID
            location: GCP region
            model_name: Vertex AI Gemini model name
            max_retries: Maximum number of retries
            retry_delay: Delay between retries
        """
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        self.location = location
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Set credentials
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if self.credentials_path and os.path.exists(self.credentials_path):
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
        else:
            self.credentials = None  # Use default credentials
            
        # Set model to None initially
        self.model = None
        
        # Initialize Vertex AI
        self._init_vertex_ai()
        
    def _init_vertex_ai(self):
        """Initialize Vertex AI client with Gemini model."""
        try:
            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials
            )
            
            # Initialize Gemini model
            self.model = GenerativeModel(self.model_name)
            
            logger.info(f"Vertex AI initialized successfully with project {self.project_id} in {self.location}")
            logger.info(f"Using Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            raise RuntimeError(f"Vertex AI initialization failed: {str(e)}")
    
    def _draw_bounding_boxes(self, image: Image.Image, bounding_boxes: List[Dict]) -> Image.Image:
        """
        Draw bounding boxes on the image for visual verification.
        
        Args:
            image: PIL Image
            bounding_boxes: List of bounding boxes in dict format
            
        Returns:
            PIL Image with bounding boxes drawn on it
        """
        # Create a copy of the image to draw on
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Image dimensions for normalization if needed
        width, height = image.size
        
        # Colors for different boxes
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Draw each bounding box
        for i, bbox in enumerate(bounding_boxes):
            # Get coordinates
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0)
            y2 = bbox.get('y2', 0)
            label = bbox.get('label', 'unlabeled')
            
            # Check if coordinates are normalized (between 0 and 1)
            # If so, convert to absolute pixel values
            if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                x1 = x1 * width
                y1 = y1 * height
                x2 = x2 * width
                y2 = y2 * height
            
            # Select color
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label if provided and not "unlabeled"
            if label and label != "unlabeled":
                # Text position above bounding box with background for better visibility
                text_position = (x1, max(0, y1 - 20))
                label_width = 6 * len(label)  # Approximate width of text
                # Draw background for text for better visibility
                draw.rectangle([text_position[0], text_position[1], 
                                text_position[0] + label_width, text_position[1] + 15], 
                               fill=(255, 255, 255))
                # Draw text
                draw.text(text_position, label, fill=color)
        
        return img_with_boxes
    
    def verify_image(self, 
                    image_bytes: bytes, 
                    conversation_text: Optional[str] = None, 
                    bounding_boxes: Optional[List[Dict]] = None) -> Tuple[LLMStatus, str]:
        """
        Verify the image content using Vertex AI.
        
        Args:
            image_bytes: Raw bytes of the image
            conversation_text: Text conversation context (optional)
            bounding_boxes: List of bounding boxes (optional)
            
        Returns:
            Tuple of (LLMStatus, explanation message)
        """
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Draw bounding boxes on image if provided
            if bounding_boxes and len(bounding_boxes) > 0:
                img = self._draw_bounding_boxes(img, bounding_boxes)
            
            # Set up the prompt with requirement for structured output
            prompt = self._create_verification_prompt(img, conversation_text, bounding_boxes)
            
            # Convert PIL image to a Part for the model
            # First convert PIL Image to a bytes buffer
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Create Part from bytes
            image_part = Part.from_data(mime_type="image/png", data=img_bytes)
            
            # Make the prediction using the GenerativeModel
            response = self.model.generate_content(
                [prompt, image_part],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                    "top_p": 0.8,
                    "top_k": 40,
                },
            )
            
            # Process the response
            result_text = response.text
            
            # Parse the result as JSON if possible
            verification_result = self._parse_verification_response(result_text)
            
            # Determine verification status based on parsed result
            if verification_result.get("verified", False):
                return LLMStatus.VERIFIED, result_text
            else:
                return LLMStatus.REJECTED, result_text
            
        except Exception as e:
            logger.error(f"Error during image verification: {str(e)}")
            return LLMStatus.REJECTED, f"Verification failed: {str(e)}"
    
    async def verify_image_async(self, 
                                image_bytes: bytes, 
                                conversation_text: Optional[str] = None, 
                                bounding_boxes: Optional[List[Dict]] = None) -> Tuple[LLMStatus, str]:
        """
        Async version of verify_image.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.verify_image,
            image_bytes,
            conversation_text,
            bounding_boxes
        )
    
    def _create_verification_prompt(self, 
                                   image: Image.Image, 
                                   conversation_text: Optional[str] = None, 
                                   bounding_boxes: Optional[List[Dict]] = None) -> str:
        """
        Create a prompt for image verification.
        
        Args:
            image: PIL Image (with bounding boxes drawn if provided)
            conversation_text: Text conversation context
            bounding_boxes: List of bounding boxes (for context only, as they're drawn on the image)
            
        Returns:
            Prompt string
        """
        prompt = "You are a meticulous AI data validation specialist for a computer vision dataset. Your task is to validate a data submission and decide if it should be 'verified' or 'rejected' based on the rules below.\n\n"
        
        # Add conversation context if available
        if conversation_text:
            prompt += f"The user provided this description: \"{conversation_text}\"\n\n"
        
        # Note that we have bounding boxes visible in the image
        if bounding_boxes and len(bounding_boxes) > 0:
            prompt += f"There are {len(bounding_boxes)} bounding boxes drawn on the image"
            labels_present = any(bbox.get('label') and bbox.get('label') != "unlabeled" for bbox in bounding_boxes)
            if labels_present:
                prompt += " with labels shown above each box.\n\n"
                # Print each label to be specific
                for i, bbox in enumerate(bounding_boxes):
                    label = bbox.get('label')
                    if label and label != "unlabeled":
                        prompt += f"  - Box {i+1} has label: '{label}'\n"
                prompt += "\n"
            else:
                prompt += " without labels.\n\n"
        
        # Add validation rules
        prompt += ("**VALIDATION RULES:**\n\n"
                  "1. **Bounding Box Accuracy:** The submission is 'verified' ONLY if ALL bounding boxes are accurately drawn "
                  "around a single, distinct object. If ANY box is inaccurate, the entire submission is 'rejected'.\n\n"
                  "2. **Label Correctness:** The submission is 'verified' ONLY if ALL "
                  "labels are contextually correct according to the **Labeling Guidelines** below. If ANY box is inaccurate OR ANY label is clearly incorrect, the entire submission is 'rejected'.\n\n"
                  "**LABELING GUIDELINES:**\n\n"
                  "* **Part-of-a-Whole Principle:** A label is considered **correct** if it identifies the whole object, even if the bounding box only covers a distinct, recognizable part of that object. The goal is to confirm the presence of the labeled object, not to granularly label its parts.\n"
                  "* **Examples of 'Verified' Labels:**\n"
                  "    * A box drawn around a **zebra's head** with the label **'zebra'** is VERIFIED.\n"
                  "    * A box drawn around a **car's wheel** with the label **'car'** is VERIFIED.\n"
                  "    * A box drawn around a **person's face** with the label **'person'** is VERIFIED.\n"
                  "* **Examples of 'Rejected' Labels:**\n"
                  "    * A box around a **zebra's head** with the label **'horse'** is REJECTED (factually incorrect).\n"
                  "    * A box around a **car's wheel** with the label **'bicycle'** is REJECTED (incorrect object).\n"
                  "    * A box without any visible object with any label is REJECTED.\n\n"
                  "3. **Additional rejection criteria:** Reject if the image contains adult content, violence, disturbing imagery, or promotes harmful activities.\n\n"
                  "4. **For multiple bounding boxes:** Evaluate each box independently and provide specific feedback for each box based on all the rules above.\n\n")
        
        # Require structured JSON output
        if bounding_boxes and len(bounding_boxes) > 1:
            # For multiple bounding boxes, request detailed evaluation of each
            prompt += ("You MUST respond with a JSON object in the following format ONLY:\n"
                      "```json\n"
                      "{\n"
                      '  "verified": true/false,\n'
                      '  "reason": "Overall explanation of your decision",\n'
                      '  "boxes": [\n'
                      '    {\n'
                      '      "box_number": 1,\n'
                      '      "label": "label_name",\n'
                      '      "is_accurate": true/false,\n'
                      '      "feedback": "Specific feedback for this box"\n'
                      '    },\n'
                      '    {\n'
                      '      "box_number": 2,\n'
                      '      "label": "label_name",\n'
                      '      "is_accurate": true/false,\n'
                      '      "feedback": "Specific feedback for this box"\n'
                      '    }\n'
                      '  ]\n'
                      "}\n"
                      "```\n\n"
                      "The 'verified' field MUST be a boolean (true/false) and should be true ONLY if ALL boxes are accurate.\n"
                      "The 'reason' field MUST be a string with your overall assessment.\n"
                      "The 'boxes' array MUST contain an entry for EACH bounding box with specific feedback.\n"
                      "For each box, include the box number, label name, whether it's accurate (true/false), and specific feedback."
                     )
        else:
            # For single or no bounding box, use simpler format
            prompt += ("You MUST respond with a JSON object in the following format ONLY:\n"
                      "```json\n"
                      "{\n"
                      '  "verified": true/false,\n'
                      '  "reason": "Explanation of your decision"\n'
                      "}\n"
                      "```\n\n"
                      "The 'verified' field MUST be a boolean (true/false) and the 'reason' field MUST be a string explaining your decision."
                     )
        
        return prompt
    

    
    def _parse_verification_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the structured JSON response from the LLM.
        
        Args:
            response_text: Text response from the LLM
            
        Returns:
            Dict containing parsed response with 'verified' (bool) and 'reason' (str) keys
        """
        # Default response
        default_response = {
            "verified": False,
            "reason": "Failed to parse response"
        }
        
        try:
            # Try to extract JSON from the response
            # Look for JSON blocks indicated by triple backticks
            if "```json" in response_text and "```" in response_text:
                # Extract content between ```json and ```
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
                result = json.loads(json_str)
                
                # Ensure we have at least the basic fields
                if "verified" not in result:
                    result["verified"] = False
                if "reason" not in result:
                    result["reason"] = "Missing verification decision"
                    
                # Log the result with potentially truncated box details for cleaner logs
                log_result = result.copy()
                if "boxes" in log_result and len(log_result["boxes"]) > 0:
                    for box in log_result["boxes"]:
                        if "feedback" in box and len(box["feedback"]) > 30:
                            box["feedback"] = box["feedback"][:30] + "..."
                
                logger.info(f"Successfully parsed JSON response: {log_result}")
                return result
            
            # Try to find any JSON-like structure in the response
            import re
            json_pattern = r'\{.*"verified":\s*(true|false).*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    potential_json = match.group(0)
                    result = json.loads(potential_json)
                    logger.info(f"Found and parsed JSON-like structure: {result}")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If we can't parse JSON but see "verified": true or similar patterns
            if '"verified": true' in response_text or '"verified":true' in response_text:
                logger.info("Found verified:true pattern in response")
                return {"verified": True, "reason": "Pattern match found in response"}
            elif '"verified": false' in response_text or '"verified":false' in response_text:
                logger.info("Found verified:false pattern in response")
                return {"verified": False, "reason": "Pattern match found in response"}
                
            # Fallback to keyword-based detection
            response_lower = response_text.lower()
            if "verified: true" in response_lower:
                return {"verified": True, "reason": "Keyword match in response"}
            elif "verified: false" in response_lower:
                return {"verified": False, "reason": "Keyword match in response"}
            
            # Last resort fallback to accept/reject keywords
            if "accept" in response_lower and "reject" not in response_lower:
                logger.warning("Falling back to keyword detection: ACCEPT found")
                return {"verified": True, "reason": "Accept keyword found"}
            elif "reject" in response_lower:
                logger.warning("Falling back to keyword detection: REJECT found")
                return {"verified": False, "reason": "Reject keyword found"}
                
            # If all attempts fail, log the response for debugging
            logger.warning(f"Failed to parse response: {response_text[:200]}...")
            return default_response
            
        except Exception as e:
            logger.error(f"Error parsing verification response: {str(e)}")
            return default_response


# Global instance
vertex_ai_client: Optional[VertexAI] = None

def get_vertex_ai() -> VertexAI:
    """Get or create the global VertexAI instance."""
    global vertex_ai_client
    if vertex_ai_client is None:
        # Get credentials path from environment
        credentials_path = os.getenv('GCS_CREDENTIALS_PATH') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        project_id = os.getenv('GCP_PROJECT_ID')
        
        if not project_id:
            # Try to extract project ID from credentials file
            if credentials_path and os.path.exists(credentials_path):
                try:
                    with open(credentials_path, 'r') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id')
                except Exception as e:
                    logger.warning(f"Failed to extract project ID from credentials: {str(e)}")
        
        if not project_id:
            raise ValueError("GCP_PROJECT_ID environment variable is required")
        
        # Initialize the client
        vertex_ai_client = VertexAI(
            credentials_path=credentials_path,
            project_id=project_id
        )
    
    return vertex_ai_client
