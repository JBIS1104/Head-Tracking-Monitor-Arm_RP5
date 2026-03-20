#!/bin/bash

# Simple Python viewer for face detection
# Uses yolodesk but displays in window instead of streaming

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL="${1:-models/yolov8n-face.pt}"
SOURCE="${2:-https://192.168.1.111:8002/mjpeg}"
CONFIDENCE="${3:-0.6}"

echo "=============================================="
echo "YOLOv8n-face Detection Viewer (Direct)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model:      $MODEL"
echo "  Source:     $SOURCE"
echo "  Confidence: $CONFIDENCE"
echo ""
echo "Note: This uses yolodesk with direct OpenCV display"
echo "      No HTTP server overhead"
echo ""
echo "Press 'q' to quit"
echo ""

# Use yolodesk but WITHOUT the --serve flag to display directly
# The --no-display flag is removed so it shows in a window
python yolodesk \
    --model "$MODEL" \
    --conf "$CONFIDENCE" \
    live \
    --source "$SOURCE" \
    --insecure \
    --max-frames 99999
