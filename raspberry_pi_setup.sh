#!/bin/bash
# Raspberry Pi Surveillance System Setup Script
# Supports Raspberry Pi 4/5 with Raspberry Pi OS (Debian-based)

echo "=========================================="
echo "Raspberry Pi Surveillance System Setup"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "[1/7] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "[2/7] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    git \
    wget

# Install Python packages
echo ""
echo "[3/7] Installing Python packages..."
pip3 install --upgrade pip

# Install PyTorch for Raspberry Pi (CPU-only, optimized)
echo ""
echo "[4/7] Installing PyTorch (this may take a while)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics YOLO
echo ""
echo "[5/7] Installing Ultralytics YOLO..."
pip3 install ultralytics

# Install additional dependencies
echo ""
echo "[6/7] Installing additional dependencies..."
pip3 install \
    numpy \
    opencv-python \
    pillow \
    pyyaml \
    requests \
    scipy \
    tqdm \
    matplotlib \
    seaborn \
    pandas

# Enable camera
echo ""
echo "[7/7] Configuring camera..."
if ! grep -q "start_x=1" /boot/config.txt 2>/dev/null; then
    echo "Enabling camera in /boot/config.txt..."
    sudo bash -c 'echo "start_x=1" >> /boot/config.txt'
    sudo bash -c 'echo "gpu_mem=128" >> /boot/config.txt'
    echo "⚠️  Camera enabled. Please reboot after setup completes."
fi

# Create project directory
echo ""
echo "Creating project directory..."
mkdir -p ~/surveillance_system
cd ~/surveillance_system

# Download model (if not already present)
if [ ! -f "best_4class_model.pt" ]; then
    echo ""
    echo "To download your trained model:"
    echo "1. Transfer from your computer using SCP:"
    echo "   scp FINAL_4CLASS_MODEL/best_4class_model.pt pi@<raspberry-pi-ip>:~/surveillance_system/"
    echo ""
    echo "2. Or download from GitHub:"
    echo "   wget https://github.com/ShivamSingh0005/yolo_trained_model_surveillance_system/raw/main/FINAL_4CLASS_MODEL/best_4class_model.pt"
fi

# Test camera
echo ""
echo "Testing camera..."
if python3 -c "import cv2; cap = cv2.VideoCapture(0); ret, _ = cap.read(); cap.release(); exit(0 if ret else 1)" 2>/dev/null; then
    echo "✓ Camera test successful"
else
    echo "⚠️  Camera test failed. Please check camera connection."
fi

# Create systemd service (optional)
echo ""
read -p "Create systemd service for auto-start? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo tee /etc/systemd/system/surveillance.service > /dev/null <<EOF
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
EOF

    sudo systemctl daemon-reload
    echo "✓ Systemd service created"
    echo "  Enable: sudo systemctl enable surveillance"
    echo "  Start:  sudo systemctl start surveillance"
    echo "  Status: sudo systemctl status surveillance"
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Transfer your model file to ~/surveillance_system/"
echo "2. Copy raspberry_pi_surveillance.py to ~/surveillance_system/"
echo "3. Run: python3 raspberry_pi_surveillance.py"
echo ""
echo "For headless operation:"
echo "  python3 raspberry_pi_surveillance.py --no-display"
echo ""
echo "For help:"
echo "  python3 raspberry_pi_surveillance.py --help"
echo ""

# Check if reboot needed
if grep -q "start_x=1" /boot/config.txt 2>/dev/null; then
    echo "⚠️  Reboot required to enable camera"
    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
fi
