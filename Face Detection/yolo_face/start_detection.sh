#!/bin/bash
# Start YOLO Face Detection from MJPEG Server Stream
# This script reads frames from https://192.168.1.111:8002/mjpeg
# and broadcasts face detection results at http://localhost:8003/

cd /home/billyp/Documents/Face\ Detection/yolo_face
source .venv/bin/activate

echo "Starting YOLO Face Detection..."
echo "Input:  https://192.168.1.111:8002/mjpeg"
echo "Output: http://localhost:8003/"
echo ""

python yolodesk live \
  --source "https://192.168.1.111:8002/mjpeg" \
  --insecure \
  --no-display \
  --serve \
  --serve-host 0.0.0.0 \
  --serve-port 8003

# To run in background:
# nohup bash start_detection.sh > yolo_detection.log 2>&1 &
