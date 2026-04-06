# 4-Class Surveillance System - YOLOv8

Production-ready surveillance system using YOLOv8n for real-time threat detection.

## 🎯 System Overview

**Performance**: 93.0% mAP@0.5 | **Status**: ✅ Production Ready

### Detected Classes
- **Animal** - 98.4% AP50
- **Forest** - 89.0% AP50
- **Militant** - 88.6% AP50
- **UAV-Drone** - 95.9% AP50

### Key Features
✅ Real-time detection (3-15 FPS on Raspberry Pi)
✅ High accuracy across all classes (88%+ AP50)
✅ Optimized for edge deployment (6.3 MB model)
✅ Automatic threat level classification
✅ Raspberry Pi support with camera integration
✅ Comprehensive logging and monitoring

## 📁 Project Structure

```
.
├── FINAL_4CLASS_MODEL/              # Production model and results
│   ├── best_4class_model.pt         # Trained model (6.3 MB)
│   ├── metrics.json                 # Performance metrics
│   ├── training_curves.png          # Training visualizations
│   ├── confusion_matrix.png         # Confusion matrix
│   └── advanced analysis figures    # ROC, state evolution, tail probability
│
├── train/                           # Training dataset (601 images)
├── valid/                           # Validation dataset (186 images)
├── test/                            # Test dataset (112 images)
├── data_4class.yaml                 # Dataset configuration
│
├── raspberry_pi_surveillance.py     # Raspberry Pi deployment script
├── advanced_analysis_4class.py      # Advanced statistical analysis
├── generate_4class_report.py        # Report generation
│
└── Documentation/
    ├── 4CLASS_SYSTEM_README.md      # System overview
    ├── RASPBERRY_PI_GUIDE.md        # Raspberry Pi deployment
    ├── DEPLOYMENT_SUMMARY.md        # All deployment options
    └── ADVANCED_ANALYSIS_4CLASS_README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ShivamSingh0005/yolo_trained_model_surveillance_system.git
cd yolo_trained_model_surveillance_system

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference (Desktop/Server)

```python
from ultralytics import YOLO

# Load model
model = YOLO('FINAL_4CLASS_MODEL/best_4class_model.pt')

# Run inference on image
results = model('image.jpg')

# Display results
results[0].show()

# Run on video
results = model('video.mp4', stream=True)
for result in results:
    result.show()
```

### 3. Raspberry Pi Deployment

```bash
# On Raspberry Pi
wget https://raw.githubusercontent.com/ShivamSingh0005/yolo_trained_model_surveillance_system/main/raspberry_pi_setup.sh
chmod +x raspberry_pi_setup.sh
./raspberry_pi_setup.sh

# Transfer model
scp FINAL_4CLASS_MODEL/best_4class_model.pt pi@<ip>:~/surveillance_system/

# Run surveillance
python3 raspberry_pi_surveillance.py
```

See [Raspberry Pi Quick Start](raspberry_pi_quickstart.md) for detailed instructions.

## 📊 Performance Metrics

### Overall Performance
- **mAP@0.5**: 92.96%
- **mAP@0.5:0.95**: 60.24%
- **Precision**: 87.48%
- **Recall**: 90.97%
- **F1-Score**: 89.19%

### Per-Class Performance

| Class | AP50 | Precision | Recall | F1-Score |
|-------|------|-----------|--------|----------|
| Animal | 98.4% | 93.5% | 90.5% | 91.9% |
| Forest | 89.0% | 74.7% | 93.8% | 83.2% |
| Militant | 88.6% | 85.4% | 84.5% | 84.9% |
| UAV-Drone | 95.9% | 96.4% | 95.2% | 95.8% |

### Advanced Analysis
- **Average AUC-ROC**: 0.856 (excellent discrimination)
- **Pre-alert Window**: 1.6 seconds for threat escalation
- **Rare Event Detection**: Strong performance at 95th percentile

See [Advanced Analysis](ADVANCED_ANALYSIS_4CLASS_README.md) for detailed statistical analysis.

## 🎯 Use Cases

### 1. Real-time Surveillance
```python
# Raspberry Pi with camera
python3 raspberry_pi_surveillance.py --camera 0

# RTSP stream
python3 raspberry_pi_surveillance.py --camera "rtsp://192.168.1.100:554/stream"

# Headless mode (no display)
python3 raspberry_pi_surveillance.py --no-display
```

### 2. Batch Processing
```python
from ultralytics import YOLO
import glob

model = YOLO('FINAL_4CLASS_MODEL/best_4class_model.pt')

# Process all images in directory
images = glob.glob('images/*.jpg')
results = model(images)

# Save results
for i, result in enumerate(results):
    result.save(f'output/result_{i}.jpg')
```

### 3. API Deployment
```python
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('FINAL_4CLASS_MODEL/best_4class_model.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    results = model(file)
    return jsonify(results[0].tojson())

app.run(host='0.0.0.0', port=5000)
```

## 🔧 Configuration

### Model Configuration
```yaml
# data_4class.yaml
train: train/images
val: valid/images
test: test/images

nc: 4  # number of classes
names: ['Animal', 'Forest', 'Militant', 'UAV-Drone']
```

### Inference Parameters
```python
# Adjust confidence threshold
results = model('image.jpg', conf=0.6)

# Adjust IoU threshold
results = model('image.jpg', iou=0.5)

# Use different image size
results = model('image.jpg', imgsz=416)

# Half precision (faster on GPU)
results = model('image.jpg', half=True)
```

## 📈 Performance Optimization

### For Raspberry Pi
```python
# Export to ONNX for faster inference
model.export(format='onnx', simplify=True)
model_onnx = YOLO('best_4class_model.onnx')

# Use lower resolution
results = model(image, imgsz=416)

# Process every Nth frame
if frame_count % 2 == 0:
    results = model(frame)
```

### For GPU
```python
# Use TensorRT (NVIDIA GPUs)
model.export(format='engine', device=0)
model_trt = YOLO('best_4class_model.engine')

# Batch processing
images = [img1, img2, img3, img4]
results = model(images)  # Process batch at once
```

## 🛠️ Development

### Generate Reports
```bash
# Generate comprehensive 4-class report
python generate_4class_report.py

# Run advanced analysis
python advanced_analysis_4class.py

# Finalize system package
python finalize_4class_system.py
```

### Test Environment
```bash
# Check environment setup
python check_environment.py

# Test Raspberry Pi setup
python test_raspberry_pi.py
```

## 📚 Documentation

### Core Documentation
- [4-Class System Overview](4CLASS_SYSTEM_README.md) - Complete system documentation
- [Deployment Guide](DEPLOYMENT_SUMMARY.md) - All deployment options
- [GitHub Setup](GITHUB_SETUP.md) - Repository management

### Raspberry Pi
- [Quick Start Guide](raspberry_pi_quickstart.md) - 5-minute setup
- [Complete Guide](RASPBERRY_PI_GUIDE.md) - Detailed documentation
- [Setup Script](raspberry_pi_setup.sh) - Automated installation

### Analysis
- [Advanced Analysis](ADVANCED_ANALYSIS_4CLASS_README.md) - Statistical analysis
- [Performance Metrics](FINAL_4CLASS_MODEL/FINAL_SUMMARY.md) - Detailed metrics

## 🔐 Security & Privacy

- Implement authentication for API endpoints
- Use HTTPS/TLS for remote access
- Encrypt stored detections
- Comply with local surveillance laws
- Implement data retention policies

See [Deployment Summary](DEPLOYMENT_SUMMARY.md) for security best practices.

## 🆘 Troubleshooting

### Common Issues

**Low FPS on Raspberry Pi**
- Use lower resolution (416x416)
- Export to ONNX format
- Add cooling (heatsink/fan)
- Process every Nth frame

**Out of Memory**
- Increase swap size
- Close other applications
- Use smaller batch size

**Camera Not Detected**
- Enable camera: `sudo raspi-config`
- Check connections: `ls /dev/video*`
- Test camera: `raspistill -o test.jpg`

**Model Not Loading**
- Verify file path
- Check file integrity
- Ensure dependencies installed

## 📊 Dataset Information

### Statistics
- **Training**: 601 images, 1,069 instances
- **Validation**: 186 images, 292 instances
- **Test**: 112 images, 184 instances
- **Total**: 899 images, 1,545 instances

### Class Distribution
- Animal: 308 instances (19.9%)
- Forest: 308 instances (19.9%)
- Militant: 308 instances (19.9%)
- UAV-Drone: 621 instances (40.2%)

## 🎓 Training Details

### Model Architecture
- **Base Model**: YOLOv8n
- **Parameters**: 3,006,428
- **Model Size**: 6.3 MB
- **Input Size**: 640x640

### Training Configuration
- **Epochs**: 100
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 → 0.0001 (cosine)
- **Augmentation**: HSV, rotation, flip, mosaic, mixup

### Training Time
- **Raspberry Pi 5**: Not recommended (use pre-trained model)
- **Desktop (RTX 3060)**: ~40 minutes
- **Desktop (CPU only)**: ~4-6 hours

## 🌟 Features

### Detection Features
- Multi-class object detection
- Real-time inference
- Automatic threat level classification
- Confidence scoring
- Bounding box visualization

### Deployment Features
- Raspberry Pi optimized
- Headless mode support
- Auto-start on boot
- Remote access (SSH/VNC)
- RTSP stream support

### Logging Features
- Detection image saving
- JSON metadata logging
- Performance statistics
- System monitoring
- Alert notifications

## 🤝 Contributing

This is a complete, production-ready system. For improvements:
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics team
- **Dataset**: Roboflow surveillance dataset
- **Framework**: PyTorch, OpenCV

## 📞 Support

- **Documentation**: See docs folder
- **Issues**: GitHub Issues
- **Repository**: https://github.com/ShivamSingh0005/yolo_trained_model_surveillance_system

## 📈 Version History

- **v1.0** (Current) - 4-class production system
  - 93% mAP@0.5
  - Raspberry Pi support
  - Complete documentation
  - Advanced analysis

---

**Status**: ✅ Production Ready | **Performance**: 93% mAP@0.5 | **Model Size**: 6.3 MB

**Last Updated**: April 2026 | **Maintained by**: ShivamSingh0005
