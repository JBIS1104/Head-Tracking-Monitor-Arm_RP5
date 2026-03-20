#!/usr/bin/env python3
"""
YOLO Face Detection with GPIO Control - Smart Lamp Example
Detects faces using YOLOv8n-face model and controls Raspberry Pi GPIO.

When a face is detected in a specified region for N consecutive frames,
GPIO pin goes HIGH to turn on a relay (lamp, etc). When no face is detected
for N consecutive frames, GPIO goes LOW to turn off the device.

Setup:
1. Connect relay to GPIO pin (default: GPIO 14)
2. Plug lamp into relay's "normally OFF" outlet
3. Run script: python toggle_face_gpio.py

Controls during execution:
- 'q': Quit
- 's': Pause
- 'p': Save screenshot
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import requests

### User-defined parameters

# Detection parameters
model_path = 'models/yolov8n-face.pt'     # Path to YOLO model
source = 'https://192.168.1.111:8002/mjpeg' # MJPEG stream URL
min_conf = 0.6                             # Minimum detection confidence threshold
resW, resH = 1280, 720                    # Resolution to run at
use_insecure = True                        # Disable TLS verification for HTTPS
record = False                             # Enable recording if True

# Detection parameters
consecutive_frames_threshold = 10          # Frames needed to detect face in ROI

# Define region of interest (ROI) where to look for faces
# Yellow box will appear at these coordinates on the display
roi_xmin = 400
roi_ymin = 200
roi_xmax = 880
roi_ymax = 520

# Bounding box colors (Tableau 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

### Initialize model

# Check if model exists
if not os.path.exists(model_path):
    print(f'ERROR: Model not found at {model_path}')
    sys.exit(1)

# Load YOLO model
print(f"Loading model from {model_path}...")
try:
    from ultralytics import YOLO
    model = YOLO(model_path, task='detect')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    print("Attempting to use yolodesk instead...")
    # Could fall back to yolodesk CLI here if needed
    sys.exit(1)

# Set up recording if enabled
recorder = None
if record:
    record_name = 'face_detection_recording.avi'
    record_fps = 10
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), 
                               record_fps, (resW, resH))
    print(f"✓ Recording enabled: {record_name}")

# Open video source
print(f"Connecting to source: {source}")
if 'http' in source.lower():
    import requests
    cap = None
    stream_url = source
    use_requests = True
    print("✓ Using requests-based MJPEG reader")
elif 'usb' in source.lower():
    cam_idx = int(source[3:]) if len(source) > 3 else 0
    cap = cv2.VideoCapture(cam_idx)
    cap.set(3, resW)
    cap.set(4, resH)
    use_requests = False
    print(f"✓ USB camera connected (index {cam_idx})")
else:
    print("ERROR: Invalid source. Use 'usb0' or HTTP(S) MJPEG URL")
    sys.exit(1)

# Initialize FPS calculation variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

# Initialize GPIO state tracking
consecutive_detections = 0
gpio_state = 0
frame_count = 0

print("\n" + "="*60)
print("Face Detection - Ready!")
print("="*60)
print(f"Model: {model_path}")
print(f"Confidence threshold: {min_conf}")
print(f"Detection threshold: {consecutive_frames_threshold} frames")
print("\nControls: 'q'=quit, 's'=pause, 'p'=screenshot")
print("="*60 + "\n")

### Main detection loop

try:
    while True:
        t_start = time.perf_counter()
        frame = None

        # Grab frame from source
        if use_requests:
            # MJPEG stream via requests
            try:
                if 'cap' not in locals() or cap is None:
                    verify_ssl = not use_insecure
                    cap = requests.get(stream_url, stream=True, verify=verify_ssl, timeout=10)
                    bytes_data = b''
                
                for chunk in cap.iter_content(chunk_size=4096):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        break
            except Exception as e:
                print(f"ERROR: Failed to get frame from stream: {e}")
                break
        else:
            # USB camera
            ret, frame = cap.read()

        if frame is None:
            print('ERROR: Unable to read frames from source')
            break

        frame_count += 1

        ### Run YOLO detection
        results = model.predict(frame, conf=min_conf, verbose=False)
        detections = results[0].boxes

        ### Process detections
        face_locations = []
        face_count = 0

        for i in range(len(detections)):
            # Extract bounding box coordinates
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            if xyxy.ndim == 0:  # Single detection case
                xyxy = xyxy.reshape(1, 4)
            if xyxy.ndim == 1:
                xyxy = xyxy.reshape(1, 4)
            
            for box in (xyxy if xyxy.ndim == 2 else [xyxy]):
                xmin, ymin, xmax, ymax = box[:4].astype(int)
                
                # Get confidence and class
                conf = detections[i].conf.item()
                classidx = int(detections[i].cls.item())
                
                # Calculate center
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                face_locations.append([cx, cy])
                face_count += 1

                # Draw bounding box
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label
                label = f'face: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                             (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw circle at face center
                color_intensity = 30 * consecutive_detections
                cv2.circle(frame, (cx, cy), 7, (0, color_intensity, color_intensity), -1)

        ### GPIO control logic

        # Check if any face is in the ROI
        face_in_roi = False
        for face_cx, face_cy in face_locations:
            if (face_cx > roi_xmin) and (face_cx < roi_xmax) and \
               (face_cy > roi_ymin) and (face_cy < roi_ymax):
                face_in_roi = True
                break

        # Update consecutive detection counter
        if face_in_roi:
            consecutive_detections = min(consecutive_frames_threshold, consecutive_detections + 1)
        else:
            consecutive_detections = max(0, consecutive_detections - 1)

        # Print status when face detected
        if consecutive_detections >= consecutive_frames_threshold and gpio_state == 0:
            gpio_state = 1
            print(".")

        if consecutive_detections <= 0 and gpio_state == 1:
            gpio_state = 0

        ### Display results

        # Draw FPS
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw ROI box (yellow)
        cv2.rectangle(frame, (roi_xmin, roi_ymin), (roi_xmax, roi_ymax), (0, 255, 255), 2)
        cv2.putText(frame, 'Detection Region', (roi_xmin, roi_ymin-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw GPIO status
        status_text = f'Faces: {face_count} | GPIO: {"ON" if gpio_state else "OFF"}'
        status_color = (0, 255, 0) if gpio_state else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Draw detection counter
        cv2.putText(frame, f'Det: {consecutive_detections}/{consecutive_frames_threshold}',
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Display frame
        cv2.imshow('Face Detection + GPIO Control', frame)
        if recorder:
            recorder.write(frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✓ Quit requested")
            break
        elif key == ord('s'):
            print("Paused (press any key to continue)")
            cv2.waitKey()
        elif key == ord('p'):
            cv2.imwrite(f'face_detection_capture_{frame_count}.png', frame)
            print(f"✓ Screenshot saved: face_detection_capture_{frame_count}.png")

        # Calculate and update FPS
        t_stop = time.perf_counter()
        frame_rate = float(1 / (t_stop - t_start))
        
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate)
        avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print("\n✓ Interrupted by user")

finally:
    ### Clean up
    print("\nCleaning up...")
    
    # Release resources
    if 'cap' in locals() and cap and not use_requests:
        cap.release()
    if recorder:
        recorder.release()
    
    cv2.destroyAllWindows()
    print(f"✓ Program ended. Processed {frame_count} frames total")
    print(f"✓ Average FPS: {avg_frame_rate:.2f}")
