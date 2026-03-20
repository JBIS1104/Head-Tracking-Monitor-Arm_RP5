# YOLO Face Detection on Raspberry Pi 5

This guide walks you through setting up YOLOv8n-Face detection on your Raspberry Pi 5 with NCNN optimization for best performance.

## Overview

- **Model**: YOLOv8n-Face (from [lindevs](https://github.com/lindevs/yolov8-face))
- **Framework**: NCNN (optimized for ARM CPUs)
- **Expected Performance**: ~15-20 FPS on Pi 5 with 640x480 resolution
- **Requirements**: 64-bit Raspberry Pi OS Bookworm

## Setup Instructions

### Step 1: Prepare Your Raspberry Pi

On your Pi terminal, ensure your system is up to date:

```bash
sudo apt update
sudo apt upgrade
```

### Step 2: Create a Virtual Environment

```bash
mkdir -p ~/yolo_face
cd ~/yolo_face
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install ultralytics ncnn opencv-python requests
```

Note: This may take 10-15 minutes on the Pi. If it stalls, press Ctrl+C and retry.

### Step 4: Download the YOLOv8n-Face Model

```bash
cd ~/yolo_face
wget https://github.com/lindevs/yolov8-face/releases/download/v1.0.1/yolov8n-face-lindevs.pt
```

Or download from your PC and transfer the file.

### Step 5: Export Model to NCNN Format

This converts the model from PyTorch to NCNN for optimal Pi performance:

```bash
yolo export model=yolov8n-face-lindevs.pt format=ncnn
```

This creates a folder named `yolov8n-face-lindevs_ncnn_model/` containing the NCNN weights.

### Step 6: Copy the Detection Script

Copy `yolo_face_detect_rpi.py` to your `~/yolo_face` directory:

```bash
# On your development machine, in the Face Detection/yolo_face folder:
# scp yolo_face_detect_rpi.py pi@<your-pi-ip>:~/yolo_face/

# Or transfer manually
```

## Usage Examples

### Live Camera Feed (USB Camera)

```bash
cd ~/yolo_face
source venv/bin/activate
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --resolution=640x480
```

Press `q` to quit.

### Different Camera

If using camera index 0 instead of usb0:
```bash
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=0 --resolution=640x480
```

### Image File

```bash
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=/path/to/image.jpg
```

### Video File

```bash
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=/path/to/video.mp4
```

### Headless Mode (No Display Window)

Useful for Pi without monitor or over SSH:

```bash
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --no-display
```

### Stream Results Over Network

Broadcast results to `http://<pi-ip>:8092/`:

```bash
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 \
  --no-display --serve --serve-port=8092
```

Access from your PC browser: `http://<pi-ip>:8092/`

### MJPEG Stream Input (from Camera or Server)

If you have an MJPEG stream source:

```bash
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model \
  --source="http://192.168.1.100:8080/mjpeg" --no-display --insecure
```

## Performance Tips

### Adjust Confidence Threshold

Lower values = more detections but more false positives:

```bash
# More sensitive (detect smaller/distant faces)
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --conf=0.3

# Less sensitive (only confident detections)
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --conf=0.7
```

### Resolution Optimization

Lower resolution = faster inference:

```bash
# Faster but less detailed
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --resolution=320x240

# Higher quality but slower
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --resolution=1280x720
```

### Background Mode (systemd Service)

To run as a background service:

1. Create `/home/pi/yolo_face/run_detector.sh`:
```bash
#!/bin/bash
cd /home/pi/yolo_face
source venv/bin/activate
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model \
  --source=usb0 --no-display --serve --serve-port=8092
```

2. Make it executable:
```bash
chmod +x /home/pi/yolo_face/run_detector.sh
```

3. Create `/etc/systemd/system/yolo-face.service`:
```ini
[Unit]
Description=YOLO Face Detection
After=network.target

[Service]
ExecStart=/home/pi/yolo_face/run_detector.sh
Restart=on-failure
RestartSec=10
User=pi

[Install]
WantedBy=multi-user.target
```

4. Enable and start:
```bash
sudo systemctl enable yolo-face
sudo systemctl start yolo-face
sudo systemctl status yolo-face
```

## Troubleshooting

### "Unable to receive frames from source"
- Check USB camera connection: `ls /dev/video*`
- Try different USB port
- Verify camera works: `ffplay /dev/video0` or `fswebcam test.jpg`

### Model export fails
```bash
# Reinstall ultralytics
pip install --upgrade --force-reinstall ultralytics
yolo export model=yolov8n-face-lindevs.pt format=ncnn
```

### Import errors
```bash
# Verify all packages
pip list | grep -E "ultralytics|ncnn|opencv"

# Reinstall if needed
pip install --upgrade ultralytics ncnn opencv-python
```

### Performance is slow
- Check system load: `top` or `htop`
- Use lower resolution: `--resolution=320x240`
- Increase confidence threshold: `--conf=0.7`
- Disable display on headless Pi: `--no-display`

### SSH and no display available

If running over SSH (no X forwarding), always use `--no-display`:

```bash
ssh pi@<your-pi-ip>
cd ~/yolo_face
source venv/bin/activate
python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --no-display --serve
```

## Optional: Use Larger Models

For better accuracy (with slower inference):

```bash
# Download YOLOv8s-Face
wget https://github.com/lindevs/yolov8-face/releases/download/v1.0.1/yolov8s-face-lindevs.pt

# Export to NCNN
yolo export model=yolov8s-face-lindevs.pt format=ncnn

# Run
python yolo_face_detect_rpi.py --model=yolov8s-face-lindevs_ncnn_model --source=usb0 --resolution=640x480
```

Available models: `yolov8n-face` (fastest), `yolov8s-face`, `yolov8m-face`, `yolov8l-face`, `yolov8x-face` (most accurate)

## References

- [YOLO on Raspberry Pi](https://www.ejtech.io/learn/yolo-on-raspberry-pi)
- [YOLOv8-Face by lindevs](https://github.com/lindevs/yolov8-face)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [NCNN Project](https://github.com/Tencent/ncnn)
