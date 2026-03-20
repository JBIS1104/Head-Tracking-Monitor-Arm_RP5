#!/bin/bash
# Quick setup script for YOLO Face Detection on Raspberry Pi 5
# Usage: bash setup_pi5.sh

set -e

echo "================================"
echo "YOLO Face Detection Setup (Pi 5)"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create workspace
echo -e "${YELLOW}Creating workspace...${NC}"
mkdir -p ~/yolo_face
cd ~/yolo_face

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "${YELLOW}Installing dependencies (this may take 10-15 minutes)...${NC}"
pip install ultralytics ncnn opencv-python requests

# Download model
echo -e "${YELLOW}Downloading YOLOv8n-Face model...${NC}"
if [ ! -f "yolov8n-face-lindevs.pt" ]; then
    wget https://github.com/lindevs/yolov8-face/releases/download/v1.0.1/yolov8n-face-lindevs.pt
    echo -e "${GREEN}Model downloaded${NC}"
else
    echo -e "${GREEN}Model already exists${NC}"
fi

# Export to NCNN
echo -e "${YELLOW}Exporting model to NCNN format (this may take 2-3 minutes)...${NC}"
if [ ! -d "yolov8n-face-lindevs_ncnn_model" ]; then
    yolo export model=yolov8n-face-lindevs.pt format=ncnn
    echo -e "${GREEN}Model exported${NC}"
else
    echo -e "${GREEN}NCNN model already exists${NC}"
fi

# Test detection
echo -e "${YELLOW}Testing detection setup...${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Connect USB camera to your Pi"
echo "2. Run: cd ~/yolo_face && source venv/bin/activate"
echo "3. Test with: python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --resolution=640x480"
echo ""
echo "For more options, see SETUP_PI5.md"
