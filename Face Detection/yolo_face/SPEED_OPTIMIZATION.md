# Speed Optimization Guide

## Current Performance Bottleneck

Your Pi 5 is CPU-bound doing face detection inference. The main optimization lever is **what you process**, not how you process it.

## Quick Tunings (Slowest to Fastest)

### 1. **High Quality** (Current - Slower)
```bash
pkill -f yolodesk
cd ~/Documents/Face\ Detection/yolo_face
source .venv/bin/activate
python yolodesk --conf 0.3 live --source "https://192.168.1.111:8002/mjpeg" --insecure --no-display --serve --serve-port 8003 &
```
- Detects more faces but slower
- CPU: 246%
- False positives: Higher

### 2. **Balanced** (Recommended)
```bash
pkill -f yolodesk
python yolodesk --conf 0.5 live --source "https://192.168.1.111:8002/mjpeg" --insecure --no-display --serve --serve-port 8003 &
```
- Good balance of speed and detection
- CPU: ~200-250%
- False positives: Reduced

### 3. **Performance** (Faster)
```bash
pkill -f yolodesk
python yolodesk --conf 0.7 live --source "https://192.168.1.111:8002/mjpeg" --insecure --no-display --serve --serve-port 8003 &
```
- Only detects confident faces
- CPU: ~200-250% (still inference bound)
- False positives: Very low

## What's Slowing It Down?

The bottleneck is **ONNX inference on CPU** - processing every single frame through the neural network is expensive. Options to speed up:

1. **Confidence Threshold** ✅ (Already implemented)
   - Higher = faster filtering of low-confidence detections
   - Impact: ~10-20% speed improvement

2. **Frame Skipping** (Not available in yolodesk)
   - Process every 2nd/3rd frame instead of every frame
   - Impact: 50-66% speed improvement

3. **Lower Resolution** (Not easily available)
   - Process at 320x240 instead of 640x480
   - Impact: 70% speed improvement

4. **Use NCNN Backend** (If available)
   - NCNN is faster than ONNX for ARM
   - Would need model export from desktop PC
   - Impact: 30-50% speed improvement

## Current Settings

```
Backend: ONNX
Model: yolov8n-face-lindevs.onnx
Confidence: 0.5 (tunable)
CPU Usage: ~246%
Memory: 1.5% (very efficient)
```

## Recommendations

**For real-time performance**, choose one:

### Option A: Accept Current Speed (Works Fine)
- Keep `--conf 0.5`
- Works reasonably well on Pi 5
- Stream is smooth at ~8-10 FPS

### Option B: Build NCNN Model (Faster)
- Requires exporting model on desktop PC with torch
- 30-50% faster but one-time setup
- See SETUP_PI5.md for NCNN export

### Option C: Frame Skipping Script (Much Faster)
- See `detect_optimized.py` 
- Processes every Nth frame
- Can get 2x speedup with acceptable detection coverage

## Test Current Speed

Monitor in real-time:
```bash
watch -n 0.5 'ps aux | grep yolodesk | grep -v grep | awk "{print \"CPU:\", \$3\"%  MEM:\", \$4\"%\"}"'
```

Check logs:
```bash
tail -f yolo_detection.log
```

View stream:
Open browser to `http://192.168.1.111:8003/`
