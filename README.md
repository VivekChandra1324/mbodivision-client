# MbodiVision Client

A Python client for interacting with the MbodiVision computer vision API. This client provides easy-to-use methods for submitting images with bounding box annotations and performing object detection.

## 🚀 **Quick Start - No Coding Required!**

### **For New Users (Easiest Way)**

#### **Step 1: Install the Client**
```bash
pip install pip install git+https://github.com/VivekChandra1324/mbodivision-client.git
```

#### **Step 2: Get the Web Interface**
```bash
# Clone the repository to get the web interface
git clone https://github.com/VivekChandra1324/mbodivision-client.git
cd mbodivision-client
pip install -e .
```

#### **Step 3: Start the Web Server**
```bash
python simple_web_server.py
```

#### **Step 4: Use Your Browser**
- Open your web browser
- Go to: **http://localhost:8000**
- **Upload images** 
- **Run object detection** automatically


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
