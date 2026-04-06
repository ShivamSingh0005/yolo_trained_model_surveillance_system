# Raspberry Pi Quick Start Guide

Get your surveillance system running in 5 minutes!

## 🚀 Super Quick Setup

### 1. Prepare Raspberry Pi (5 minutes)

```bash
# SSH into your Raspberry Pi
ssh pi@<raspberry-pi-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-opencv git

# Install Python packages
pip3 install ultralytics opencv-python numpy
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Transfer Files (2 minutes)

**From your computer:**

```bash
# Transfer model
scp FINAL_4CLASS_MODEL/best_4class_model.pt pi@<raspberry-pi-ip>:~/

# Transfer surveillance script
scp raspberry_pi_surveillance.py pi@<raspberry-pi-ip>:~/
```

### 3. Run Surveillance (1 minute)

**On Raspberry Pi:**

```bash
# Test setup
python3 test_raspberry_pi.py

# Run surveillance
python3 raspberry_pi_surveillance.py
```

**Press 'q' to quit, 's' to save frame**

## 📊 What to Expect

- **FPS**: 3-15 depending on your Pi model
- **Detection**: Real-time threat classification
- **Output**: Saved to `raspberry_pi_detections/`

## 🎯 Common Commands

```bash
# Headless mode (no display)
python3 raspberry_pi_surveillance.py --no-display

# Higher confidence threshold
python3 raspberry_pi_surveillance.py --conf 0.7

# Use USB camera
python3 raspberry_pi_surveillance.py --camera 1

# RTSP stream
python3 raspberry_pi_surveillance.py --camera "rtsp://192.168.1.100:554/stream"
```

## 🔧 Troubleshooting

### Camera not working?
```bash
# Enable camera
sudo raspi-config
# Interface Options -> Camera -> Enable

# Reboot
sudo reboot
```

### Low FPS?
```bash
# Use lower resolution
# Edit raspberry_pi_surveillance.py line 60:
# self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
# self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
```

### Out of memory?
```bash
# Increase swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 📖 Full Documentation

See `RASPBERRY_PI_GUIDE.md` for complete documentation including:
- Auto-start on boot
- Email/Telegram alerts
- Performance optimization
- Remote access setup

## ✅ Success Checklist

- [ ] Raspberry Pi updated
- [ ] Python packages installed
- [ ] Model file transferred
- [ ] Camera enabled and working
- [ ] Test script passes
- [ ] Surveillance running

## 🆘 Need Help?

1. Run test script: `python3 test_raspberry_pi.py`
2. Check full guide: `RASPBERRY_PI_GUIDE.md`
3. GitHub issues: https://github.com/ShivamSingh0005/yolo_trained_model_surveillance_system/issues

---

**Ready to deploy? Let's go! 🚀**
