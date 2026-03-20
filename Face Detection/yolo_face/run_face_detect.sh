#!/bin/bash

# Quick start script for YOLOv8n-face detection on Raspberry Pi 5
# Uses pre-trained model from WIDERFace dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "YOLOv8n-face Detection - Quick Start"
echo "Pre-trained on WIDERFace dataset"
echo "=================================="
echo ""

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Error: .venv not found. Run setup first."
    exit 1
fi

# Check model exists
if [ ! -f "models/yolov8n-face.pt" ]; then
    echo "Error: Model not found at models/yolov8n-face.pt"
    echo "Run: wget https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov8n-face.pt -O models/yolov8n-face.pt"
    exit 1
fi

echo "✓ Model found: models/yolov8n-face.pt (6.1 MB)"
echo ""

# Parameters
MJPEG_URL="${1:-https://192.168.1.111:8002/mjpeg}"
CONFIDENCE="${2:-0.6}"
OUTPUT_PORT="${3:-8003}"

echo "Configuration:"
echo "  Source:      $MJPEG_URL"
echo "  Confidence:  $CONFIDENCE"
echo "  Output port: $OUTPUT_PORT"
echo ""
echo "Run detection with:"
echo "  python detect_widerface.py --source \"$MJPEG_URL\" --conf $CONFIDENCE --port $OUTPUT_PORT --insecure"
echo ""
echo "View results at: http://localhost:$OUTPUT_PORT/"
echo ""

# Run detection
python detect_widerface.py \
    --source "$MJPEG_URL" \
    --conf "$CONFIDENCE" \
    --port "$OUTPUT_PORT" \
    --insecure
