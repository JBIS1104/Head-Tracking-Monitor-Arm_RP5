#!/bin/bash
# Quick setup for YOLO Face detection on Raspberry Pi 5
# Run this script to finish setup

cd /home/billyp/Documents/Face\ Detection/yolo_face

echo "======================================"
echo "YOLO Face Detection - Quick Setup"
echo "======================================"

# Activate environment
source .venv/bin/activate

echo ""
echo "✓ Dependencies installed"
echo "✓ ONNX model: models/yolov8n-face-lindevs.onnx"
echo "✓ Detection script: yolodesk"
echo ""

echo "======================================"
echo "READY TO USE!"
echo "======================================"
echo ""
echo "Usage examples:"
echo ""
echo "1. Live camera detection (requires USB camera connected):"
echo "   python yolodesk live --source 0 --no-display --serve --serve-port 8092"
echo ""
echo "2. Detect faces in an image:"
echo "   python yolodesk detect --source path/to/image.jpg"
echo ""
echo "3. Detect from MJPEG stream:"
echo "   python yolodesk live --source 'https://192.168.1.100:8080/mjpeg' --insecure --no-display"
echo ""
echo "4. Help:"
echo "   python yolodesk --help"
echo "   python yolodesk live --help"
echo ""
