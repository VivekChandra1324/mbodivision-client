# MbodiVision Client

> **A Complete MLOps Platform for YOLO Object Detection with Automated Training Pipeline**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-red.svg)](https://fastapi.tiangolo.com/)
[![Async](https://img.shields.io/badge/async-supported-brightgreen.svg)](https://docs.python.org/3/library/asyncio.html)

MbodiVision Client is a comprehensive, production-ready MLOps platform that democratizes computer vision by providing an end-to-end solution for YOLO object detection. It combines data collection, automated training, model deployment, and inference capabilities in a single, easy-to-use package with both programmatic access and a user-friendly web interface.

## Key Features
- **Zero-Configuration MLOps**: Complete automated training pipeline with no manual intervention.
- **AI-Powered Data Validation**: Uses Google Vertex AI (Gemini) for intelligent label verification.
- **Cloud-Native Architecture**: Built for Google Cloud Platform with auto-scaling capabilities.
- **Production-Ready**: Includes monitoring, logging, error handling, and state persistence.
- **Developer-Friendly**: Clean APIs, comprehensive documentation, and extensive testing.
- **YOLO-Optimized**: Specialized for YOLO models with automatic dataset conversion and training.

## Project Architecture Overview

The MbodiVision Client is built with a modular, scalable architecture that separates concerns while maintaining tight integration:

```
mbodivision-client/
├── mbodivision_client/             # Core Python Package (Installable)
│   ├── __init__.py                 # Package initialization & exports
│   ├── client.py                   # Async HTTP client implementation
│   └── models.py                   # Pydantic data models & validation
├── app/                            # Full-Featured Web Application
│   ├── templates/                  # HTML templates (upload.html, inference.html)
│   ├── main.py                     # Production FastAPI server
│   ├── cloud_storage.py            # Google Cloud Storage integration
│   ├── training_pipeline.py        # Automated training pipeline
│   ├── yolo_service.py             # YOLO inference service
│   ├── llm_verification.py         # AI-powered data validation
│   └── vertex_ai.py                # Google Vertex AI integration
├── simple_web_server.py            # Lightweight demo server
├── examples/                       # Developer code examples
├── tests/                          # Test suite
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
└── README.md                       # This comprehensive guide
```

### **System Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MbodiVision MLOps Platform                   │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Web UI)          │  Backend (FastAPI)                │
│  ├─ Upload Interface        │  ├─ Data Collection API           │
│  ├─ Inference Interface     │  ├─ YOLO Service                  │
│  └─ Real-time Feedback      │  └─ Training Pipeline Manager     │
├─────────────────────────────────────────────────────────────────┤
│  AI Services                    Cloud Infrastructure            │
│  ├─ LLM Verification        │  ├─ Google Cloud Storage          │
│  ├─ Vertex AI Integration   │  ├─ PostgreSQL Database           │
│  └─ Model Evaluation        │  └─ Vertex AI Training            │
├─────────────────────────────────────────────────────────────────┤
│  Data Flow & Processing                                         │
│  Image Upload → Validation → Storage → Training → Deployment    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start 

### Option 1: Web Interface

#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/mbodivision/mbodivision-client.git
cd mbodivision-client
```

#### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **Step 3: Configure Environment**
```bash
cp env.example .env
# Edit .env with your Google Cloud credentials
```

#### **Step 4: Start the Web Server**
```bash
python simple_web_server.py
```

#### **Step 5: Use Your Browser**
- Open: **http://localhost:8000**
- **Upload images** with bounding box annotations
- **Run object detection** with real-time results

### **Option 2: Python Client Library**

#### **Install the Package**
```bash
pip install git+https://github.com/mbodivision/mbodivision-client.git
```

#### **Basic Usage**
```python
import asyncio
from mbodivision_client import MbodiVisionClient, BoundingBox

async def main():
    async with MbodiVisionClient("http://localhost:8000") as client:
        # Method 1: Using BoundingBox objects
        boxes = [
            BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5, label="person"),
            BoundingBox(x1=0.6, y1=0.2, x2=0.9, y2=0.7, label="car")
        ]
        
        result = await client.submit_data(
            image_path="test_image.jpg",
            bounding_boxes=boxes,
            conversation="Training data for person and car detection"
        )
        print(f"Submission ID: {result.submission_id}")
        
        # Method 2: Using dictionary format (simplified)
        bbox_dict = {
            "person": (0.1, 0.2, 0.8, 0.9),  # (x1, y1, x2, y2)
            "car": (0.3, 0.4, 0.7, 0.6)
        }
        
        response = await client.submit_data_with_dict_bbox(
            image_path="image.jpg",
            bbox_dict=bbox_dict,
            conversation="A person walking near a car"
        )
        
        # Run inference
        result = await client.detect_objects("test_image.jpg", conf=0.5)
        print(f"Detected {len(result.detections)} objects")

asyncio.run(main())
```



## Core Components Deep Dive

### **1. Data Collection & Validation (`app/main.py`)**
- **FastAPI-based REST API** with automatic OpenAPI documentation
- **Multi-format image support** (JPEG, PNG) with automatic conversion
- **Real-time bounding box validation** with pixel-to-normalized coordinate conversion
- **Asynchronous processing** for high-throughput data ingestion
- **Background LLM verification** using Google Vertex AI

### **2. AI-Powered Data Quality (`app/llm_verification.py`, `app/vertex_ai.py`)**
- **Intelligent label verification** using Google Gemini 2.0 Flash
- **Context-aware validation** considering image content and conversation
- **Automated quality scoring** with detailed feedback
- **Configurable validation rules** for different use cases
- **Fallback mechanisms** for robust operation

### **3. Cloud Storage & Database (`app/cloud_storage.py`)**
- **Google Cloud Storage integration** for scalable image storage
- **PostgreSQL database** for metadata and pipeline state management
- **Automatic data persistence** with retry logic and error handling
- **Async/await patterns** for non-blocking I/O operations
- **Data integrity checks** and cleanup mechanisms

### **4. Automated Training Pipeline (`app/training_pipeline.py`)**
- **Event-driven training** triggered by data threshold (configurable)
- **Batch processing** with intelligent data splitting (train/validation)
- **Model versioning** with automatic best-model promotion
- **Persistent state management** for crash recovery
- **Metrics tracking** (mAP50-95, precision, recall)

### **5. YOLO Service (`app/yolo_service.py`)**
- **Cloud-model inference** with automatic model downloading
- **Real-time object detection** with configurable confidence thresholds
- **Model caching** for improved performance
- **Automatic model refresh** when better models are available
- **Error handling** and graceful degradation

### **6. Dataset Management (`app/dataset_converter.py`)**
- **YOLO format conversion** with automatic class mapping
- **Train/validation splitting** with reproducible random seeds
- **Cloud-native dataset creation** without local storage
- **Concurrent processing** for large datasets
- **Metadata preservation** throughout the pipeline

### **7. Cloud Training (`app/cloud_training.py`)**
- **Vertex AI integration** for scalable GPU training
- **Custom container support** for YOLO training
- **Automatic artifact detection** (best.pt, results.csv)
- **Job monitoring** with timeout and retry logic
- **Resource optimization** with configurable machine types

## 🔧 **Configuration & Customization**

### **Environment Variables**
```bash
# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1

# Storage Configuration
GCS_BUCKET_NAME=your-bucket-name
POSTGRES_URL=postgresql://user:pass@host:port/db

# Training Configuration
TRAINING_THRESHOLD=5  # Minimum images to trigger training
AUTO_START_PIPELINE=true

# Model Configuration
YOLO_CONTAINER_URI=gcr.io/project/yolo-training:latest
```


### **REST API Endpoints**

#### **Data Submission Endpoint**
```http
POST /api/upload
Content-Type: multipart/form-data

Parameters:
- image: File (required) - Image file (JPEG/PNG)
- bounding_boxes_json: String (required) - JSON array of bounding boxes
- conversation: String (optional) - Context description

Response:
{
  "submission_id": "uuid",
  "message": "Data received successfully. LLM verification is in progress."
}
```

#### **Object Detection Endpoint**
```http
POST /api/yolo/detect
Content-Type: multipart/form-data

Parameters:
- image: File (required) - Image file for detection
- conf: Float (optional) - Confidence threshold (default: 0.3)

Response:
{
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.95,
      "x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9
    }
  ],
  "annotated_image": "base64_encoded_image",
  "model_name": "gs://bucket/model.pt"
}
```

## Testing & Development

### **Quick Tests**
```bash
# Test client installation
python -c "from mbodivision_client import MbodiVisionClient; print('✅ Client works!')"

# Test web server
python simple_web_server.py
# Visit http://localhost:8000

# Run test suite
python -m pytest tests/

# Development mode with auto-reload
python app/main.py  # Full featured server
```

### **Production Checklist**
- [ ] Add authentication and authorization
- [ ] Validate and sanitize all inputs
- [ ] Implement rate limiting
- [ ] Use HTTPS in production
- [ ] Set up logging and monitoring
- [ ] Configure CORS properly
- [ ] Add API key management




## Acknowledgments

- **Ultralytics** for the YOLO framework
- **Google Cloud Platform** for cloud infrastructure
- **FastAPI** for the web framework
- **Pydantic** for data validation




