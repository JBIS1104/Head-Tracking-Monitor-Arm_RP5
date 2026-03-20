# Toggle Face Detection GPIO Control - Setup Guide

## Overview

This script detects faces using YOLOv8n-face model and controls Raspberry Pi GPIO pins. It's a face detection version of the EdjeElectronics smart lamp example.

**Use Case**: Automatically turn on/off a lamp, relay, or other GPIO-controlled device when a face is detected in a specific region.

---

## Hardware Setup

### Required:
- Raspberry Pi 5 (or Pi 4)
- Relay module (like DIGITAL LOGGERS IoT AC Power Relay)
- GPIO jumper wires
- Device to control (lamp, fan, etc.)

### Wiring (GPIO 14):
```
Raspberry Pi GPIO 14 → Relay Input → Relay Control
Relay "Normally OFF" outlet → Lamp or device
```

Default GPIO pin: **GPIO 14** (configurable in script)

---

## Software Setup

### 1. Install GPIO Library

```bash
pip install gpiozero
```

Without `gpiozero`, script runs in simulation mode (no actual GPIO control).

### 2. Make Script Executable

```bash
chmod +x toggle_face_gpio.py
```

### 3. Run the Script

```bash
cd "/home/billyp/Documents/Face Detection/yolo_face"
python toggle_face_gpio.py
```

---

## Configuration

Edit the user-defined parameters at the top of `toggle_face_gpio.py`:

### Detection Parameters
```python
model_path = 'models/yolov8n-face.pt'              # Model file path
source = 'https://192.168.1.111:8002/mjpeg'       # Video source (MJPEG URL or 'usb0')
min_conf = 0.6                                     # Confidence threshold
resW, resH = 1280, 720                            # Resolution
use_insecure = True                                # Ignore SSL cert (for self-signed)
record = False                                     # Enable video recording
```

### GPIO Parameters
```python
gpio_pin = 14                              # GPIO pin number
consecutive_frames_threshold = 10          # Frames to hold before toggling
```

### Region of Interest (ROI)
```python
# Yellow box shows where to look for faces
roi_xmin = 400
roi_ymin = 200
roi_xmax = 880
roi_ymax = 520
```

**To find ROI coordinates**:
1. Run the script
2. Note where you want the yellow box
3. Adjust coordinates until box is in desired location
4. Update values in script

---

## Controls During Execution

| Key | Action |
|-----|--------|
| **q** | Quit program |
| **s** | Pause (press any key to resume) |
| **p** | Save screenshot |

---

## Display Output

The window shows:
- **Real-time video** with face detection boxes
- **Yellow box**: Region of Interest (where faces are monitored)
- **Green circles**: Detected faces (intensity shows detection count)
- **GPIO Status**: Current relay state (ON/OFF)
- **Detection Counter**: How many consecutive frames face detected in ROI
- **FPS**: Real-time processing speed

---

## How It Works

### Logic Flow:

1. **Read frame** from MJPEG stream or USB camera
2. **Detect faces** using YOLOv8n model
3. **Check if face in ROI** (yellow box region)
4. **If face detected**:
   - Increment detection counter
   - When counter ≥ threshold → Turn GPIO HIGH (Relay ON)
5. **If no face detected**:
   - Decrement detection counter
   - When counter ≤ 0 → Turn GPIO LOW (Relay OFF)
6. **Display results** with all information

### Example Timeline:

```
Frame 1:  Face in ROI → counter = 1
Frame 2:  Face in ROI → counter = 2
...
Frame 10: Face in ROI → counter = 10 → GPIO ON (threshold reached)
Frame 11: No face     → counter = 9
...
Frame 20: No face     → counter = 0 → GPIO OFF
```

---

## Troubleshooting

### "GPIO not available"
- Install gpiozero: `pip install gpiozero`
- Script will run in simulation mode if unavailable

### "Failed to connect to source"
- Check MJPEG URL is accessible
- Verify SSL certificate issue with `--insecure` flag
- Try USB camera instead: `source = 'usb0'`

### "No faces detected"
- Lower confidence threshold: `min_conf = 0.4`
- Ensure faces are visible in stream
- Check ROI coordinates - make yellow box covers target area

### "Faces detected but GPIO not toggling"
- Verify GPIO library installed: `pip install gpiozero`
- Check GPIO pin number (default: 14)
- Test relay connection manually
- Check consecutive_frames_threshold (default: 10)

### "False positives (relay keeps toggling)"
- Increase confidence threshold: `min_conf = 0.8`
- Increase consecutive_frames_threshold: `20` or higher
- Adjust ROI to ignore false positive areas

---

## Input Sources

### MJPEG Stream (Network)
```python
source = 'https://192.168.1.111:8002/mjpeg'
use_insecure = True  # For self-signed certificates
```

### USB Camera
```python
source = 'usb0'  # or 'usb1', 'usb2', etc.
```

### Local Video File (testing)
```python
source = 'test_video.mp4'
```

---

## Output Files

When running, script may create:
- `face_detection_recording.avi` - Video recording (if `record = True`)
- `face_detection_capture_*.png` - Screenshots (press 'p' during run)

---

## Performance

Expected performance on Raspberry Pi 5:
- **FPS**: 8-12 FPS (depending on resolution)
- **CPU**: 150-180%
- **Memory**: <2%
- **GPIO Response**: Immediate (0.1s)

---

## Examples

### Smart Lamp (Default Setup)
```python
gpio_pin = 14
consecutive_frames_threshold = 10
# Lamp turns on when face detected for 10 frames in ROI
```

### Sensitive Detection (Quick On/Off)
```python
consecutive_frames_threshold = 3
min_conf = 0.7
# Faster response, stricter confidence
```

### Strict Detection (Reliable)
```python
consecutive_frames_threshold = 20
min_conf = 0.8
# Slower but very reliable, reduces false positives
```

---

## Advanced Usage

### Multiple GPIO Pins
To control multiple devices, create multiple LED instances:
```python
led1 = gpiozero.LED(14)  # Device 1
led2 = gpiozero.LED(15)  # Device 2

# Later in code:
if condition1:
    led1.on()
if condition2:
    led2.on()
```

### Custom Relay Logic
Modify the GPIO control section to implement custom logic:
```python
# Example: Blink pattern instead of steady on
if face_in_roi:
    if frame_count % 10 == 0:
        led.toggle()  # Blink every 10 frames
```

### Logging Events
Record when GPIO toggles:
```python
with open('gpio_events.log', 'a') as f:
    f.write(f"[{time.time()}] GPIO ON - Face detected\n")
```

---

## Notes

- Script must be run with GPIO permissions (may need `sudo`)
- GPIO pins are zero-indexed (GPIO 14 = physical pin 8+6th)
- **Safety**: Never connect high voltage directly to GPIO
- Always use relay module for AC devices
- Test relay connection before running script

---

## Comparison with EdjeElectronics Example

| Feature | EdjeElectronics | This Script |
|---------|-----------------|------------|
| **Detection** | Person detection | Face detection |
| **Model** | YOLOv8 person | YOLOv8n-face |
| **Source** | USB/Picamera | USB/MJPEG stream |
| **GPIO Pin** | 14 (configurable) | 14 (configurable) |
| **Display** | cv2.imshow() | cv2.imshow() |
| **ROI** | Rectangle box | Rectangle box |

---

## References

- EdjeElectronics example: https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/tree/main/examples/toggle_pi_gpio
- YOLOv8 Face Detection: https://github.com/akanametov/yolo-face
- gpiozero documentation: https://gpiozero.readthedocs.io/
- Raspberry Pi GPIO pins: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html

---

## License

Based on EdjeElectronics example. See original repository for license.
