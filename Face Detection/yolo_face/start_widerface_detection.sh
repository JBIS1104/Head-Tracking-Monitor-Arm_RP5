#!/bin/bash

# Start YOLOv8n-face detection using pre-trained WIDERFace model
# This script uses yolodesk (already installed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration (can be overridden via command line args)
MODEL="${1:-models/yolov8n-face.pt}"
SOURCE="${2:-https://192.168.1.111:8002/mjpeg}"
CONFIDENCE="${3:-0.6}"
PORT="${4:-8003}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}YOLOv8n-face Detection${NC}"
echo -e "${GREEN}Pre-trained on WIDERFace${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check model exists
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: Model not found at $MODEL${NC}"
    echo "Available models:"
    ls -lh models/*.pt 2>/dev/null || echo "  No .pt models found"
    echo ""
    exit 1
fi

# Show config
MODEL_SIZE=$(du -h "$MODEL" | cut -f1)
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:      $MODEL ($MODEL_SIZE)"
echo "  Source:     $SOURCE"
echo "  Confidence: $CONFIDENCE"
echo "  Output:     http://localhost:$PORT/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run detection (note: --model and --conf go BEFORE the 'live' command)
python yolodesk \
    --model "$MODEL" \
    --conf "$CONFIDENCE" \
    live \
    --source "$SOURCE" \
    --insecure \
    --no-display \
    --serve \
    --serve-port "$PORT" \
    --serve-host "0.0.0.0"
