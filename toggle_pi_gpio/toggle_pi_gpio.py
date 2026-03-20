# YOLO detection script on Picamera

import os
import sys
import argparse
import glob
import time
import gpiozero
import threading

import cv2
import numpy as np
from urllib3.exceptions import InsecureRequestWarning
import requests
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from ultralytics import YOLO

### Set user-defined parameters and program parameters

# User-defined parameters
model_path = 'models/yolov8n-face_ncnn_model'    # NCNN face detection model folder
cam_source = 'https://192.168.1.111:8002/mjpeg'  # MJPEG stream URL (or 'usb0' / 'picamera0')
min_thresh = 0.5 					# Minimum detection threshold
resW, resH = 640, 480				# Resolution (balanced for speed: ~4 FPS vs 1.4 FPS at 1280x720)
record = False						# Enables recording if True

# Program parameters
# Define Raspberry Pi GPIO pin to toggle
gpio_pin = 14

# Define box coordinates where we want to look for a person. If a person is present in this box for enough frames, toggle GPIO to turn light on.
pbox_xmin = 540
pbox_ymin = 160
pbox_xmax = 760
pbox_ymax = 450

# Set detection bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

### Initialize YOLO model, GPIO, and camera

# Set up Raspberry Pi GPIO
led = gpiozero.LED(gpio_pin)

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Set up recording
if record:
    record_name = 'demo6.avi'
    record_fps = 5
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Initialize Picamera, USB camera, or MJPEG stream depending on user input
if 'http' in cam_source:
    cam_type = 'mjpeg'
    print(f'Initializing MJPEG stream: {cam_source}')
    
    # Shared frame buffer (background thread writes, main thread reads)
    mjpeg_frame_buffer = {'frame': None, 'lock': threading.Lock()}
    mjpeg_stop_event = threading.Event()
    
    def mjpeg_reader_thread():
        """Background thread: read MJPEG stream and decode frames."""
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries and not mjpeg_stop_event.is_set():
            try:
                session = requests.Session()
                # Disable SSL verification for self-signed cert, max 2s to connect
                response = session.get(cam_source, stream=True, verify=False, timeout=(2, 5))
                response.raise_for_status()
                print(f'[MJPEG] Connected, reading frames...')
                retry_count = 0
                mjpeg_buffer = b''
                last_end_pos = 0
                
                for chunk in response.iter_content(chunk_size=32768):  # 32KB chunks (4x larger)
                    if mjpeg_stop_event.is_set():
                        break
                    mjpeg_buffer += chunk
                    
                    # Optimized: Only search from where we left off, looking for end marker first
                    while True:
                        # Find JPEG start from last_end_pos
                        jpg_start = mjpeg_buffer.find(b'\xff\xd8', last_end_pos)
                        if jpg_start == -1:
                            # No complete frame yet, trim old data
                            if len(mjpeg_buffer) > 512*1024:
                                # Keep last 256KB
                                mjpeg_buffer = mjpeg_buffer[-256*1024:]
                                last_end_pos = 0
                            break
                        
                        # Find JPEG end after the start marker
                        jpg_end = mjpeg_buffer.find(b'\xff\xd9', jpg_start + 2)
                        if jpg_end == -1:
                            # End marker not found yet, need more data
                            break
                        
                        # Extract complete JPEG
                        jpg = mjpeg_buffer[jpg_start:jpg_end+2]
                        last_end_pos = jpg_end + 2
                        
                        # Decode JPEG
                        decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if decoded is not None and decoded.size > 0:
                            # Resize only if needed (most servers send correct res already)
                            h, w = decoded.shape[:2]
                            if (h, w) != (resH, resW):
                                decoded = cv2.resize(decoded, (resW, resH), interpolation=cv2.INTER_LINEAR)
                            with mjpeg_frame_buffer['lock']:
                                mjpeg_frame_buffer['frame'] = decoded
                        
                        # Continue looking for next frame
                            
            except requests.exceptions.Timeout:
                retry_count += 1
                print(f'[MJPEG] Timeout (retry {retry_count}/{max_retries}, waiting 1s...)')
                time.sleep(1)
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                print(f'[MJPEG] Connection error (retry {retry_count}/{max_retries})')
                time.sleep(1)
            except Exception as e:
                retry_count += 1
                print(f'[MJPEG] Error: {e} (retry {retry_count}/{max_retries})')
                time.sleep(1)
        
        if retry_count >= max_retries:
            print(f'[MJPEG] Failed to connect after {max_retries} retries.')
    
    # Start background thread
    reader = threading.Thread(target=mjpeg_reader_thread, daemon=True)
    reader.start()
    print('Connected to MJPEG stream (background thread active)')

elif 'usb' in cam_source:
    cam_type = 'usb'
    cam_idx = int(cam_source[3:])
    cam = cv2.VideoCapture(cam_idx)
    ret = cam.set(3, resW)
    ret = cam.set(4, resH)

elif 'picamera' in cam_source:
    from picamera2 import Picamera2
    cam_type = 'picamera'
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cam.start()

else:
    print('Invalid input for cam_source variable! Use "usb0" or "picamera0". Exiting program.')
    sys.exit()


# Initialize frame rate variables 
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

# Initialize control and status variables
consecutive_detections = 0
gpio_state = 0

# For MJPEG, wait for first frame with timeout
if cam_type == 'mjpeg':
    print('[MAIN] Waiting for first frame from MJPEG stream (max 10s)...')
    frame_wait_start = time.time()
    while time.time() - frame_wait_start < 10:
        with mjpeg_frame_buffer['lock']:
            if mjpeg_frame_buffer['frame'] is not None:
                print('[MAIN] First frame received! Starting inference loop.')
                break
        time.sleep(0.1)
    else:
        print('[ERROR] No frames received from MJPEG stream in 10 seconds.')
        print(f'[ERROR] Check that {cam_source} has a camera client sending frames.')
        sys.exit(1)

### Begin main inference loop
while True:

    t_start = time.perf_counter()

    # Grab frame from camera or MJPEG stream
    if cam_type == 'mjpeg':
        # Grab latest frame from background thread buffer (non-blocking)
        with mjpeg_frame_buffer['lock']:
            frame = mjpeg_frame_buffer['frame']
        
        if frame is None:
            # Still waiting for first frame from stream
            time.sleep(0.01)
            continue

    elif cam_type == 'usb':
        ret, frame = cam.read()

    elif cam_type == 'picamera':
        frame_bgra = cam.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR) # Remove alpha channel

    # Check to make sure frame was received
    if (frame is None):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    ### Run inference on frame and parse detections
    
    # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize array to hold locations of "person" detections
    person_locations = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
        
        # Calculate center coordinates from xyxy coordinates
        cx = int((xmin + xmax)/2)
        cy = int((ymin + ymax)/2)

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.5:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # If this object is a face, append their coordinates to running list of face detections
            if classname in ('person', 'face'):
                person_locations.append([cx, cy])
                # Draw a cirle there too (and make it change color based on number of consecutive detections)
                color_intensity = 30*consecutive_detections
                cv2.circle(frame, (cx, cy), 7, (0,color_intensity,color_intensity), -1) 

    ### Logic to trigger GPIO change
    
    # Initialize flag to indicate whether face is in the desired location this frame (set as False)
    person_in_pbox = False
    
    # Go through face detections to check if any are within desired box location
    for person_xy in person_locations:
        
        person_cx, person_cy = person_xy # Get center coordinates for this face
        
        # This big conditional checks if the face's center_x/center_y coordinates are within the box coordinates
        if (person_cx > pbox_xmin) and (person_cx < pbox_xmax) and (person_cy > pbox_ymin) and (person_cy < pbox_ymax):
            person_in_pbox = True

    # If there is a person in the box, increment consecutive detection count by 1 (but not above 15)
    if person_in_pbox == True:
        consecutive_detections = min(8, consecutive_detections + 1) # Prevents this variable from going above 15 
    
    # If not, decrease consecutive detection count by 1 (but not below 0)
    else:
        consecutive_detections = max(0, consecutive_detections - 1)
    
    # If consecutive detections are high enough AND the GPIO is currently off, turn GPIO on!
    if consecutive_detections >= 8 and gpio_state == 0:
        gpio_state = 1
        led.on() # Sets GPIO pin to HIGH (3.3V) state
    
    # Conversely, if consecutive detections are back to 0 AND the GPIO is currently on, turn GPIO off!
    if consecutive_detections <= 0 and gpio_state == 1:
        gpio_state = 0
        led.off() # Sets GPIO pin to LOW (0V) state
    
    
    ### Display results

    # Draw framerate
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    
    # Draw rectangle around the detection box where we are looking for a person
    cv2.rectangle(frame, (pbox_xmin, pbox_ymin), (pbox_xmax, pbox_ymax), (0,255,255), 2)
    
    # Draw GPIO status on frame
    if gpio_state == 0:
        cv2.putText(frame, 'Light currently OFF.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2)
    elif gpio_state == 1:
        cv2.putText(frame, 'Face detected in box! Turning light ON.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 3)
        cv2.putText(frame, 'Face detected in box! Turning light ON.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    # Display detection results
    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # Wait 5ms before moving to next frame and check for user keypress.
    key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if record: recorder.release()
if cam_type == 'mjpeg':
    mjpeg_stop_event.set()
    reader.join(timeout=2)
if cam_type == 'usb': cam.release()
if cam_type == 'picamera': cam.stop()
cv2.destroyAllWindows()
