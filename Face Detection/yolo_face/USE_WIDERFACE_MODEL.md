# Using YOLOv8n-face (Pre-trained on WIDERFace)

You now have the pre-trained YOLOv8n-face model ready to use on your Raspberry Pi 5!

## Model Files

- **models/yolov8n-face.pt** (6.1 MB) - PyTorch format for training/conversion
- **models/yolov8n-face.onnx** (13 MB) - ONNX format for inference

## Quick Start

### Option 1: Using yolodesk CLI (Simplest)

```bash
cd ~/Documents/Face\ Detection/yolo_face

# Run detection on MJPEG stream
# Note: --model and --conf go BEFORE the 'live' command
python yolodesk \
    --model models/yolov8n-face.pt \
    --conf 0.6 \
    live \
    --source "https://192.168.1.111:8002/mjpeg" \
    --insecure \
    --no-display \
    --serve --serve-port 8003

# View results at: http://192.168.1.111:8003/stream.mjpeg
```

### Option 2: Using ONNX Model (Faster)

```bash
python yolodesk \
    --model models/yolov8n-face.onnx \
    --conf 0.6 \
    live \
    --source "https://192.168.1.111:8002/mjpeg" \
    --insecure \
    --no-display \
    --serve --serve-port 8003
```

### Option 3: Using Custom Detection Script

If you have ultralytics installed in a separate environment:

```bash
python detect_widerface.py \
    --source "https://192.168.1.111:8002/mjpeg" \
    --model models/yolov8n-face.pt \
    --conf 0.6 \
    --port 8003 \
    --insecure
```

## Model Details

| Property | Value |
|----------|-------|
| Model | YOLOv8n-face |
| Training Dataset | WIDERFace |
| Task | Face Detection |
| Size (PT) | 6.1 MB |
| Size (ONNX) | 13 MB |
| Expected FPS on Pi 5 | 8-10 FPS |
| Expected CPU Usage | ~150% (multi-threaded) |
| Confidence Threshold | 0.6 (adjust 0.3-0.8) |

## Inference Commands

### On a single image:

```bash
python yolodesk \
    --model models/yolov8n-face.pt \
    --conf 0.6 \
    detect \
    --source test_image.jpg
```

### On a video file:

```bash
python yolodesk \
    --model models/yolov8n-face.pt \
    --conf 0.6 \
    live \
    --source video.mp4
```

### On live USB camera:

```bash
python yolodesk \
    --model models/yolov8n-face.pt \
    --conf 0.6 \
    live \
    --source 0
```

## Tuning Performance

### Faster inference (lower accuracy):
- Increase confidence threshold: `--conf 0.7` or `--conf 0.8`
- Use smaller model: (already using nano - the smallest)

### Better accuracy (slower):
- Decrease confidence threshold: `--conf 0.4` or `--conf 0.5`
- Higher resolution: `--imgsz 800` (default 640)

## Comparison with Other Models

Your previous setup used:
- `yolov8n-face-lindevs.onnx` (custom fine-tuned model)

This new model uses:
- `yolov8n-face.pt` (pre-trained on WIDERFace - larger diverse dataset)

**Benefits of WIDERFace model:**
- Better generalization (trained on ~32K images from WIDERFace)
- Handles various face sizes, poses, and lighting
- More robust detection in different conditions

## Troubleshooting

### Command not found: "yolodesk"
The yolodesk executable might not be in your PATH. Try:
```bash
python yolodesk ...
```
Instead of just `yolodesk ...`

### Model loading slow on first run
This is normal - the first inference loads the model (~10-20 seconds). Subsequent runs are fast.

### CPU usage too high
Increase confidence threshold:
```bash
--conf 0.7  # instead of 0.6
```

This filters low-confidence detections before drawing.

### No faces detected
1. Check confidence threshold is not too high
2. Verify stream is working: `curl -k https://192.168.1.111:8002/mjpeg | head -c 1000`
3. Test with image first: `python yolodesk predict --model models/yolov8n-face.pt --source test.jpg`

## Next Steps

1. **Run detection:** Use one of the commands above
2. **Monitor performance:** Watch CPU/FPS in terminal
3. **Fine-tune:** Adjust `--conf` threshold based on results
4. **Deploy:** Set up auto-start script (see below)

## Auto-start on Boot (Optional)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/face-detection.service
```

Add:
```ini
[Unit]
Description=YOLOv8n-face Detection Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=billyp
WorkingDirectory=/home/billyp/Documents/Face Detection/yolo_face
ExecStart=/usr/bin/python3 yolodesk live --model models/yolov8n-face.pt --source "https://192.168.1.111:8002/mjpeg" --insecure --conf 0.6 --serve --serve-port 8003
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable face-detection
sudo systemctl start face-detection
```

Monitor:
```bash
sudo systemctl status face-detection
sudo journalctl -u face-detection -f
```

## Resources

- **Repository:** https://github.com/akanametov/yolo-face
- **Releases:** https://github.com/YapaLab/yolo-face/releases/tag/1.0.0
- **WIDERFace Dataset:** http://shuoyang1213.me/WIDERFACE/
- **yolodesk CLI:** Pre-installed in your setup
