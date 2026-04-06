# Raspberry Pi Deployment Guide

Complete guide for deploying the 4-class YOLO surveillance system on Raspberry Pi 4/5.

## 📋 Requirements

### Hardware
- **Raspberry Pi 4 (4GB+ RAM)** or **Raspberry Pi 5** (recommended)
- **Camera**: Raspberry Pi Camera Module v2/v3 or USB webcam
- **Storage**: 32GB+ microSD card (Class 10 or better)
- **Power**: Official Raspberry Pi power supply (5V 3A for Pi 4, 5V 5A for Pi 5)
- **Optional**: Heatsink/fan for better performance

### Software
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Python**: 3.8+
- **Model**: best_4class_model.pt (6.3 MB)

## 🚀 Quick Start

### 1. Setup Raspberry Pi

```bash
# Download setup script
wget https://raw.githubusercontent.com/ShivamSingh0005/yolo_trained_model_surveillance_system/main/raspberry_pi_setup.sh

# Make executable
chmod +x raspberry_pi_setup.sh

# Run setup (takes 15-30 minutes)
./raspberry_pi_setup.sh
```

### 2. Transfer Model File

**Option A: Using SCP (from your computer)**
```bash
scp FINAL_4CLASS_MODEL/best_4class_model.pt pi@<raspberry-pi-ip>:~/surveillance_system/
```

**Option B: Using USB Drive**
```bash
# Mount USB drive
sudo mount /dev/sda1 /mnt/usb

# Copy model
cp /mnt/usb/best_4class_model.pt ~/surveillance_system/

# Unmount
sudo umount /mnt/usb
```

**Option C: Download from GitHub**
```bash
cd ~/surveillance_system
wget https://github.com/ShivamSingh0005/yolo_trained_model_surveillance_system/raw/main/FINAL_4CLASS_MODEL/best_4class_model.pt
```

### 3. Transfer Python Script

```bash
# From your computer
scp raspberry_pi_surveillance.py pi@<raspberry-pi-ip>:~/surveillance_system/
```

### 4. Run Surveillance

```bash
cd ~/surveillance_system
python3 raspberry_pi_surveillance.py
```

## 📖 Usage

### Basic Usage

```bash
# Run with default settings (camera 0, confidence 0.5)
python3 raspberry_pi_surveillance.py

# Custom confidence threshold
python3 raspberry_pi_surveillance.py --conf 0.6

# Use USB camera
python3 raspberry_pi_surveillance.py --camera 1

# Use RTSP stream
python3 raspberry_pi_surveillance.py --camera "rtsp://192.168.1.100:554/stream"
```

### Headless Mode (No Display)

```bash
# Run without display (for SSH/remote operation)
python3 raspberry_pi_surveillance.py --no-display

# Don't save detections
python3 raspberry_pi_surveillance.py --no-save

# Process limited frames (for testing)
python3 raspberry_pi_surveillance.py --max-frames 100
```

### Command Line Options

```
--model PATH          Path to YOLO model (default: FINAL_4CLASS_MODEL/best_4class_model.pt)
--conf FLOAT          Confidence threshold 0-1 (default: 0.5)
--camera ID/URL       Camera ID or RTSP URL (default: 0)
--no-display          Run without display (headless mode)
--no-save             Do not save detections
--max-frames N        Maximum frames to process
```

## 🎯 Performance Optimization

### 1. Model Optimization

For better performance on Raspberry Pi, you can use a smaller model or optimize the existing one:

```python
from ultralytics import YOLO

# Load model
model = YOLO('best_4class_model.pt')

# Export to ONNX (faster inference)
model.export(format='onnx', simplify=True)

# Use ONNX model
model = YOLO('best_4class_model.onnx')
```

### 2. Resolution Optimization

Lower resolution = faster processing:

```python
# In raspberry_pi_surveillance.py, modify camera settings:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)   # Lower resolution
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
```

### 3. Frame Skipping

Process every Nth frame:

```python
# Add to run() method:
if frame_count % 2 == 0:  # Process every 2nd frame
    results, detections = self.detect_threats(frame)
```

### 4. CPU Optimization

```bash
# Increase GPU memory
sudo nano /boot/config.txt
# Add: gpu_mem=256

# Overclock (Pi 4 only, use with caution)
# Add to /boot/config.txt:
# over_voltage=6
# arm_freq=2000
```

## 📊 Expected Performance

| Raspberry Pi Model | Resolution | FPS | Notes |
|-------------------|------------|-----|-------|
| Pi 4 (4GB) | 640x480 | 3-5 | Recommended |
| Pi 4 (8GB) | 640x480 | 4-6 | Better multitasking |
| Pi 5 (4GB) | 640x480 | 8-12 | Best performance |
| Pi 5 (8GB) | 640x480 | 10-15 | Optimal |

## 🔧 Troubleshooting

### Camera Not Detected

```bash
# Check camera connection
vcgencmd get_camera

# Should show: supported=1 detected=1

# Test camera
raspistill -o test.jpg

# For USB camera
ls /dev/video*
```

### Low FPS

1. **Lower resolution**: Use 416x416 or 320x320
2. **Reduce confidence threshold**: Use 0.6 or higher
3. **Use ONNX model**: Export and use optimized format
4. **Close other applications**: Free up CPU/RAM
5. **Add cooling**: Heatsink or fan prevents thermal throttling

### Out of Memory

```bash
# Increase swap size
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Import Errors

```bash
# Reinstall dependencies
pip3 install --upgrade ultralytics opencv-python numpy

# Check installations
python3 -c "import cv2; import torch; from ultralytics import YOLO; print('OK')"
```

## 🔄 Auto-Start on Boot

### Method 1: Systemd Service (Recommended)

```bash
# Create service file
sudo nano /etc/systemd/system/surveillance.service
```

Add:
```ini
[Unit]
Description=Raspberry Pi Surveillance System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/surveillance_system
ExecStart=/usr/bin/python3 /home/pi/surveillance_system/raspberry_pi_surveillance.py --no-display
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable surveillance
sudo systemctl start surveillance

# Check status
sudo systemctl status surveillance

# View logs
sudo journalctl -u surveillance -f
```

### Method 2: Crontab

```bash
crontab -e

# Add:
@reboot sleep 30 && cd /home/pi/surveillance_system && /usr/bin/python3 raspberry_pi_surveillance.py --no-display >> /home/pi/surveillance.log 2>&1
```

## 📁 Output Structure

```
raspberry_pi_detections/
├── images/
│   ├── detection_20240406_143022.jpg
│   ├── detection_20240406_143045.jpg
│   └── ...
└── logs/
    ├── detection_20240406_143022.json
    ├── detection_20240406_143045.json
    └── ...
```

### Log Format

```json
{
  "timestamp": "2024-04-06T14:30:22.123456",
  "detections": [
    {
      "class": "Militant",
      "confidence": 0.87,
      "bbox": [120, 80, 340, 420],
      "threat_level": "HIGH",
      "timestamp": "2024-04-06T14:30:22.123456"
    }
  ],
  "image_path": "raspberry_pi_detections/images/detection_20240406_143022.jpg"
}
```

## 🌐 Remote Access

### VNC (with Display)

```bash
# Enable VNC
sudo raspi-config
# Interface Options -> VNC -> Enable

# Connect from computer using VNC Viewer
# Address: <raspberry-pi-ip>:5900
```

### SSH (Headless)

```bash
# From your computer
ssh pi@<raspberry-pi-ip>

# Run surveillance
cd ~/surveillance_system
python3 raspberry_pi_surveillance.py --no-display

# View detections
ls -lh raspberry_pi_detections/images/
```

### Web Interface (Optional)

Create a simple Flask web server to view detections:

```python
# web_viewer.py
from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    images = os.listdir('raspberry_pi_detections/images')
    return render_template('index.html', images=images)

@app.route('/images/<filename>')
def image(filename):
    return send_from_directory('raspberry_pi_detections/images', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🔐 Security Considerations

1. **Change default password**: `passwd`
2. **Enable firewall**: `sudo ufw enable`
3. **Disable SSH password auth**: Use SSH keys
4. **Update regularly**: `sudo apt update && sudo apt upgrade`
5. **Secure camera stream**: Use HTTPS/VPN for remote access

## 📊 Monitoring

### System Resources

```bash
# CPU temperature
vcgencmd measure_temp

# CPU usage
htop

# Memory usage
free -h

# Disk usage
df -h
```

### Detection Statistics

```bash
# Count detections
ls raspberry_pi_detections/images/ | wc -l

# View recent logs
tail -f raspberry_pi_detections/logs/*.json

# Analyze detections
python3 -c "
import json
import glob

logs = glob.glob('raspberry_pi_detections/logs/*.json')
classes = {}
for log in logs:
    with open(log) as f:
        data = json.load(f)
        for det in data['detections']:
            cls = det['class']
            classes[cls] = classes.get(cls, 0) + 1

print('Detection Summary:')
for cls, count in sorted(classes.items()):
    print(f'  {cls}: {count}')
"
```

## 🎓 Advanced Features

### Email Alerts

```python
# Add to raspberry_pi_surveillance.py
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def send_alert(self, frame, detections):
    if not any(d['threat_level'] == 'HIGH' for d in detections):
        return
    
    msg = MIMEMultipart()
    msg['Subject'] = 'HIGH THREAT DETECTED'
    msg['From'] = 'surveillance@example.com'
    msg['To'] = 'alert@example.com'
    
    # Add text
    text = f"Detected {len(detections)} threats at {datetime.now()}"
    msg.attach(MIMEText(text))
    
    # Add image
    _, buffer = cv2.imencode('.jpg', frame)
    image = MIMEImage(buffer.tobytes())
    msg.attach(image)
    
    # Send
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your-email@gmail.com', 'your-app-password')
        server.send_message(msg)
```

### Telegram Bot

```python
import requests

def send_telegram_alert(self, frame, detections):
    bot_token = 'YOUR_BOT_TOKEN'
    chat_id = 'YOUR_CHAT_ID'
    
    message = f"🚨 {len(detections)} threats detected!"
    
    # Send message
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    requests.post(url, data={'chat_id': chat_id, 'text': message})
    
    # Send image
    _, buffer = cv2.imencode('.jpg', frame)
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    files = {'photo': buffer.tobytes()}
    requests.post(url, data={'chat_id': chat_id}, files=files)
```

## 📚 Additional Resources

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [OpenCV Raspberry Pi Guide](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

## 🆘 Support

For issues or questions:
1. Check GitHub Issues: https://github.com/ShivamSingh0005/yolo_trained_model_surveillance_system/issues
2. Review troubleshooting section above
3. Check Raspberry Pi forums

## 📝 License

This project uses the trained YOLO model for surveillance purposes. Ensure compliance with local laws regarding surveillance and privacy.

---

**Status**: ✅ Production Ready for Raspberry Pi 4/5
**Performance**: 3-15 FPS depending on hardware
**Model Size**: 6.3 MB (optimized for edge deployment)
