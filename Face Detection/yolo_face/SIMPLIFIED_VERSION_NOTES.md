# Face Detection Simplified Version - Modifications

## Changes Made

The `toggle_face_gpio.py` script has been simplified to remove GPIO control and replace it with simple console output.

### What Was Removed:
1. ✓ `gpiozero` library import and initialization
2. ✓ GPIO pin setup code (`gpiozero.LED(gpio_pin)`)
3. ✓ GPIO state tracking and control logic (`led.on()`, `led.off()`)
4. ✓ All GPIO-related print statements about relay activation

### What Was Changed:
1. **GPIO Control → Print Output**: When a face is detected in the ROI for the threshold duration, the script now prints "." instead of toggling a relay
2. **Input Source**: Uses MJPEG stream from `https://192.168.1.111:8002/mjpeg` (with SSL verification disabled for self-signed certificate)
3. **Output Display**: Face detections display on local monitor via `cv2.imshow()`

### Configuration (Lines 32-48):
```python
model_path = 'models/yolov8n-face.pt'        # YOLOv8n-face model
source = 'https://192.168.1.111:8002/mjpeg'  # MJPEG stream
min_conf = 0.6                                # Confidence threshold
consecutive_frames_threshold = 10             # Frames to confirm detection
```

### Region of Interest (ROI):
- Yellow box on display at coordinates (400,200) to (880,520)
- Face must be detected in this region for N consecutive frames
- Prints "." when threshold reached

## How to Run:
```bash
cd /home/billyp/Documents/Face\ Detection/yolo_face
python toggle_face_gpio.py
```

## Output:
- **Display**: OpenCV window showing MJPEG stream with detected faces and ROI box
- **Console**: Prints "." when face detected in ROI (threshold reached)
- **Controls**: 'q' to quit, 's' to pause, 'p' for screenshot

## Features Still Working:
- ✓ Real-time face detection with YOLOv8n-face
- ✓ MJPEG stream input at 10 FPS
- ✓ Region of Interest (ROI) filtering
- ✓ Confidence-based filtering (0.6 minimum)
- ✓ FPS display and performance monitoring
- ✓ Screenshot capability
- ✓ Face bounding boxes with colors

## Removed Features:
- ✗ GPIO relay control
- ✗ gpiozero library dependency
- ✗ GPIO state tracking
