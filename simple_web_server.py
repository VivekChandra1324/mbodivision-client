#!/usr/bin/env python3
"""
Simple web server for MbodiVision - just serves the web interface.
No complex dependencies, just upload and inference pages.
"""

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import json
from typing import Optional

# Create FastAPI app
app = FastAPI(
    title="MbodiVision Web Interface",
    description="Simple web interface for image upload and object detection",
    version="1.0.0"
)

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Serve static files
app.mount("/static", StaticFiles(directory="app"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/inference", response_class=HTMLResponse)
async def inference(request: Request):
    """Inference page for object detection."""
    return templates.TemplateResponse("inference.html", {"request": request})

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "MbodiVision web server is running!"}

@app.post("/api/upload")
async def handle_upload(
    image: UploadFile = File(...),
    bounding_boxes_json: str = Form(...),
    conversation: Optional[str] = Form(None)
):
    """Handle image upload with bounding boxes."""
    try:
        # For demo purposes, just return success
        # In a real implementation, you'd process the image
        return {
            "submission_id": "demo_" + str(hash(image.filename)),
            "message": "Image uploaded successfully! (Demo mode)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/yolo/detect")
async def detect_objects(
    image: UploadFile = File(...),
    conf: float = Form(0.3)
):
    """Handle object detection requests."""
    try:
        # For demo purposes, return sample detections
        # In a real implementation, you'd run YOLO detection
        return {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 300]
                },
                {
                    "class_name": "car",
                    "confidence": 0.87,
                    "bbox": [300, 200, 500, 400]
                }
            ],
            "annotated_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            "model_name": "demo_model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting MbodiVision Web Server...")
    print("=" * 50)
    print("📱 Web interface will be available at:")
    print("   📤 Upload page: http://localhost:8000/")
    print("   🔍 Inference page: http://localhost:8000/inference")
    print("   💚 Health check: http://localhost:8000/health")
    print("\n💡 Instructions:")
    print("   1. Open your web browser")
    print("   2. Go to http://localhost:8000")
    print("   3. Upload an image and draw bounding boxes")
    print("   4. Or go to /inference for object detection")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Fix: Use the app object directly instead of import string
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
