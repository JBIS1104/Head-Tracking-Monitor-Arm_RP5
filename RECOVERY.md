# Pi 64-bit OS Recovery Guide

After flashing Raspberry Pi OS 64-bit (Bookworm), follow these steps to restore everything.

---

## 1. Copy Your Code Back

All code lives in `/home/billyp/Documents/`. Copy it back from your backup (USB drive or other machine).

Key folders to restore:
- `Face Detection/`
- `Face recognition/`
- `Server/`
- `Servo Testing/`
- `toggle_pi_gpio/`
- `Yolo desktop/`
- `CLAUDE.md`, `RECOVERY.md`

---

## 2. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y

# OpenCV system dependencies
sudo apt install -y python3-opencv python3-pip python3-venv

# GPIO
sudo apt install -y python3-lgpio python3-gpiozero

# SSL (for Flask https)
sudo apt install -y python3-openssl
```

---

## 3. Face Detection + Monitor Arm Tracking (MAIN PROJECT)

This is the core: YOLO face detection via raw NCNN, reading from MJPEG stream.

### Create venv

```bash
cd "/home/billyp/Documents/Face Detection"
python3 -m venv .venv-gui --system-site-packages
source .venv-gui/bin/activate
```

### Install packages

```bash
pip install ncnn requests gpiozero lgpio
```

> **Do NOT** `pip install opencv-python` — use the system one (`--system-site-packages` handles it)

### Verify everything works

```bash
python -c "
import platform; print('arch:', platform.architecture())   # must say 64bit
import ncnn;    print('ncnn:', ncnn.__version__)
import cv2;     print('cv2:', cv2.__version__)
import numpy;   print('numpy:', numpy.__version__)
import requests; print('requests: ok')
import gpiozero; print('gpiozero: ok')
"
```

Expected output:
```
arch: ('64bit', 'ELF')
ncnn: 1.0.20260114  (or newer)
cv2: 4.x.x
numpy: 1.x.x
requests: ok
gpiozero: ok
```

### The NCNN model

Already in the repo at:
```
toggle_pi_gpio/models/yolov8n-face_ncnn_model/
  model.ncnn.param
  model.ncnn.bin
```
This is the YOLOv8n-face WIDERFace model — no re-download needed.

### Run face detection + tracking

```bash
source "/home/billyp/Documents/Face Detection/.venv-gui/bin/activate"
cd /home/billyp/Documents/toggle_pi_gpio

# Test without GPIO (no hardware connected)
python monitor_arm_track.py --no-gpio --insecure

# Full run with GPIO (servos + actuator)
python monitor_arm_track.py --insecure
```

Expected FPS on 64-bit: **5-8 FPS** (up from 3 FPS on 32-bit)

### Tuning parameters

```bash
# More smoothing / slower response
python monitor_arm_track.py --ema 0.6 --kp 0.25 --kd 0.15 --insecure

# Faster response / less smoothing
python monitor_arm_track.py --ema 0.3 --kp 0.45 --kd 0.08 --insecure
```

---

## 4. MJPEG Server

The server receives frames from the iPad camera and broadcasts them as an MJPEG stream.

### Create venv

```bash
cd /home/billyp/Documents/Server
python3 -m venv .venv
source .venv/bin/activate
pip install flask pyopenssl
```

### Start server

```bash
source /home/billyp/Documents/Server/.venv/bin/activate
cd /home/billyp/Documents/Server
python -c "from mjpeg_server.app import app; app.run(host='0.0.0.0', port=8002, threaded=True, ssl_context='adhoc')"
```

Stream is available at: `https://192.168.1.111:8002/mjpeg`
Check status: `https://192.168.1.111:8002/status`

---

## 5. Hardware GPIO Pin Reference

| Component | GPIO Pin | Notes |
|-----------|----------|-------|
| Yaw servo (horizontal pan) | 18 | PWM, 400Hz |
| Roll servo (vertical tilt) | 13 | PWM, 400Hz |
| Linear actuator DIR (HR8833) | 20 | H-bridge direction |
| Linear actuator EN (HR8833) | 21 | H-bridge enable |
| GPIO LED (toggle test) | 14 | toggle_pi_gpio.py |

---

## 6. Network Config

| Item | Address |
|------|---------|
| Pi IP | 192.168.1.111 |
| MJPEG server | https://192.168.1.111:8002/mjpeg |
| iPad camera app pushes frames to | https://192.168.1.111:8002/upload |

Make sure the Pi keeps the same IP (set a static IP or DHCP reservation on your router).

---

## 7. Face Recognition (separate project)

```bash
cd "/home/billyp/Documents/Face recognition"
python3 -m venv .venv
source .venv/bin/activate
pip install dlib==19.24.4 face-recognition numpy Pillow requests pigpio
```

> Note: `dlib` takes a long time to compile (~30 min on Pi). Let it run.

Trained face encodings are saved at:
```
Face recognition/output/encodings.pkl
```
If you backed this up, copy it back — no need to retrain.

---

## 8. Quick Start After Recovery

```bash
# Terminal 1: Start MJPEG server
source /home/billyp/Documents/Server/.venv/bin/activate
cd /home/billyp/Documents/Server
python -c "from mjpeg_server.app import app; app.run(host='0.0.0.0', port=8002, threaded=True, ssl_context='adhoc')"

# Terminal 2: Start face detection + tracking
source "/home/billyp/Documents/Face Detection/.venv-gui/bin/activate"
cd /home/billyp/Documents/toggle_pi_gpio
python monitor_arm_track.py --insecure
```
