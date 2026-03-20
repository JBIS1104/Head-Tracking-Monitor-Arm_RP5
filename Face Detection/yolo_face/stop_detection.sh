#!/bin/bash
# Stop YOLO Face Detection process

pkill -f "yolodesk live --source"
echo "YOLO Face Detection stopped"
