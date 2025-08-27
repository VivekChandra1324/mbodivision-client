# MbodiVision Client

A Python client for interacting with the MbodiVision computer vision API. This client provides easy-to-use methods for submitting images with bounding box annotations and performing object detection.

## 🚀 **Quick Start - No Coding Required!**

### **For New Users (Easiest Way)**

#### **Step 1: Install the Client**
```bash
pip install git+https://github.com/yourusername/mbodivision-client.git
```

#### **Step 2: Get the Web Interface**
```bash
# Clone the repository to get the web interface
git clone https://github.com/yourusername/mbodivision-client.git
cd mbodivision-client
```

#### **Step 3: Start the Web Server**
```bash
python simple_web_server.py
```

#### **Step 4: Use Your Browser**
- Open your web browser
- Go to: **http://localhost:8000**
- **Upload images** and **draw bounding boxes** with your mouse
- **Run object detection** automatically
- **No coding required!**

---

## 💻 **For Developers (Optional)**

### **Install and Use in Python Code**
```bash
# Install the client
pip install git+https://github.com/yourusername/mbodivision-client.git
```

```python
import asyncio
from mbodivision_client import MbodiVisionClient, BoundingBox

async def main():
    async with MbodiVisionClient(base_url="http://localhost:8000") as client:
        # Upload image with bounding boxes
        response = await client.submit_data_with_dict_bbox(
            image_path="my_image.jpg",
            bbox_dict={"person": (0.1, 0.2, 0.8, 0.9)},
            conversation="A person in the image"
        )
        print(f"Uploaded! ID: {response.submission_id}")
        
        # Run object detection
        result = await client.detect_objects("my_image.jpg", conf=0.5)
        print(f"Found {len(result.detections)} objects")

asyncio.run(main())
```

---

## 🌐 **Web Interface Features**

### **📤 Upload Page (http://localhost:8000/)**
- **Drag & Drop Images**: Simply drag your image files
- **Visual Bounding Box Editor**: Draw boxes around objects with your mouse
- **Easy Labeling**: Type labels like "person", "car", "cat"
- **Description**: Add text about what's in the image
- **One-Click Submit**: Upload everything with a single button

### **🔍 Inference Page (http://localhost:8000/inference)**
- **Image Upload**: Select any image for object detection
- **Confidence Control**: Adjust how certain the AI should be
- **Instant Results**: See detected objects with confidence scores
- **Annotated Images**: View images with boxes drawn around objects

---

## 📋 **What You Need**

### **Requirements**
- **Python 3.8+** installed on your computer
- **pip** (Python package installer)
- **Web browser** (Chrome, Firefox, Safari, etc.)

### **What You Get**
- ✅ **Client Package**: Install with pip
- ✅ **Web Interface**: Upload images and draw bounding boxes
- ✅ **Object Detection**: AI-powered object recognition
- ✅ **Easy to Use**: No programming knowledge required

---

## 🔧 **Installation Methods**

### **Method 1: From GitHub (Recommended)**
```bash
pip install git+https://github.com/yourusername/mbodivision-client.git
```

### **Method 2: From Local Repository**
```bash
git clone https://github.com/yourusername/mbodivision-client.git
cd mbodivision-client
pip install -e .
```

---

## 📁 **Repository Structure**

```
mbodivision-client/
├── mbodivision_client/     # Client package (pip installable)
├── app/                    # Web interface templates
├── simple_web_server.py    # Web server startup script
├── examples/               # Code examples for developers
├── tests/                  # Test suite
├── pyproject.toml          # Package configuration
├── setup.py                # Alternative installation
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🎯 **User Scenarios**

### **Scenario 1: New User (No Coding Required)** ✅
1. **Install**: `pip install git+https://github.com/yourusername/mbodivision-client.git`
2. **Clone**: `git clone https://github.com/yourusername/mbodivision-client.git`
3. **Start**: `python simple_web_server.py`
4. **Use**: Open http://localhost:8000 in your browser
5. **Upload & Annotate**: Through the web interface

### **Scenario 2: Developer**
1. **Install**: Same pip command
2. **Import**: `from mbodivision_client import MbodiVisionClient`
3. **Use in Code**: All the Python examples above

### **Scenario 3: Production**
1. **Install**: Same pip command
2. **Configure**: Set up environment variables
3. **Deploy**: Point to production server
4. **Scale**: Handle multiple users and batch processing

---

## 🧪 **Testing**

```bash
# Test the client installation
python -c "from mbodivision_client import MbodiVisionClient; print('✅ Client works!')"

# Run the test suite
python -m pytest tests/

# Test the web server
python simple_web_server.py
# Then open http://localhost:8000 in your browser
```

---

## 📞 **Support & Help**

### **Getting Started**
1. **Follow the Quick Start** section above
2. **Use the web interface** - it's designed to be simple
3. **Check error messages** - they usually tell you what to fix

### **Common Issues**
- **Port 8000 in use**: Change the port in `simple_web_server.py`
- **Image not loading**: Check file format (.jpg, .png, .jpeg)
- **Server not starting**: Make sure you're in the right directory

### **Need More Help?**
- **Web Interface**: Use the browser - no coding needed!
- **Code Issues**: Check the examples in the `examples/` folder
- **GitHub Issues**: Create an issue on the repository

---

## 🎉 **You're Ready!**

### **For New Users:**
- ✅ **Install with pip**: One command
- ✅ **Start server**: One command
- ✅ **Use web interface**: Just open your browser
- ✅ **Upload & detect**: All through the web interface

### **For Developers:**
- ✅ **Client library**: Full Python API
- ✅ **Async support**: High-performance operations
- ✅ **Type safety**: Complete type hints
- ✅ **Production ready**: Handle real workloads

**Start with the web interface - it's the easiest way to get started!** 🚀

---

## 📝 **Next Steps**

1. **Try the web interface** first (easiest)
2. **Explore the code examples** if you want to program
3. **Customize for your needs** using the client library
4. **Deploy to production** when you're ready

**The client is designed to grow with you - start simple and scale up!** 🎯
