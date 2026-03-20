# YOLO Face Detection Setup Complete ✓

Your Raspberry Pi 5 is now ready for face detection!

## What's Installed

✅ **yolodesk** - Face detection script with multiple backends  
✅ **ONNX Model** - yolov8n-face-lindevs.onnx (12 MB, optimized for inference)  
✅ **Dependencies** - ncnn, opencv-python, requests  
✅ **Python venv** - Isolated environment ready to go  

## Quick Start

### 1. **Live Camera Detection** (requires USB camera)

First, plug in your USB camera, then:

```bash
cd ~/Documents/Face\ Detection/yolo_face
source .venv/bin/activate
python yolodesk live --source 0 --no-display --serve --serve-port 8092
```

Then open in your browser: `http://<pi-ip>:8092/`

### 2. **List Available Cameras**

```bash
ls /dev/video*
```

If your camera is at `/dev/video19` instead of `/dev/video0`:

```bash
python yolodesk live --source 19 --no-display --serve
```

### 3. **Test with an Image**

```bash
python yolodesk detect --source path/to/image.jpg
```

### 4. **Run from MJPEG Stream** (like your server)

```bash
python yolodesk live --source "https://192.168.1.111:8002/mjpeg" --insecure --no-display --serve
```

## Command Options

```bash
# Show all options
python yolodesk --help
python yolodesk live --help

# Key options:
#   --source CAMERA/URL       Camera index, video file, or stream URL
#   --conf THRESHOLD          Confidence (default: 0.25, range: 0-1)
#   --no-display             Run without GUI window
#   --serve                  Broadcast results at /stream.mjpeg
#   --serve-port PORT        Custom port for MJPEG output
#   --insecure               Skip HTTPS certificate check
#   --max-frames N           Limit frames processed
```

## Current Setup Status

| Item | Status |
|------|--------|
| Python venv | ✅ Active |
| ONNX Backend | ✅ Working |
| OpenCV | ✅ v4.10.0.84 |
| NCNN | ✅ Installed |
| Model | ✅ yolov8n-face-lindevs.onnx (12 MB) |
| Camera | ⚠️  Not detected (plug in USB camera) |

## Troubleshooting

### "Cannot open video source: 0"
- Camera isn't connected or recognized
- Try different indices: `--source 1`, `--source 19`, etc.
- Check: `ls /dev/video*`

### Slow Performance
- Reduce resolution: `--resolution 320x240`
- Increase confidence threshold: `--conf 0.5`
- Use `--no-display` to skip GUI rendering

### HTTPS Errors
- Add `--insecure` flag for self-signed certificates

### Permission Denied on /dev/video*
```bash
sudo usermod -a -G video $USER
# Then logout and login
```

## Performance Notes

- **YOLOv8n-Face** on Pi 5: ~15-20 FPS with ONNX backend
- Model uses CPU only (no GPU acceleration needed)
- ONNX backend is faster than PyTorch for inference

## Environment Info

```
Python: 3.11
OS: Raspberry Pi OS (64-bit)
Architecture: ARM64
venv location: .venv/
```

## Next Steps

1. **Plug in USB camera** to Pi USB port
2. **Run**: `python yolodesk live --source 0 --no-display --serve --serve-port 8092`
3. **Access stream** from PC browser at `http://<pi-ip>:8092/`
4. **Customize** confidence, resolution, and other options as needed

## Integration with Your Server

The MJPEG stream from yolodesk can be served on your local network:

```bash
python yolodesk live --source 0 --serve --serve-host 0.0.0.0 --serve-port 8092
```

Then from another Pi/device:

```bash
python yolodesk live --source "http://192.168.1.100:8092/stream.mjpeg" --no-display --serve --serve-port 8093
```

## References

- [yolodesk GitHub](https://github.com/lindevs/yolov8-face)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [Raspberry Pi Camera Setup](https://www.raspberrypi.com/documentation/accessories/camera.html)
