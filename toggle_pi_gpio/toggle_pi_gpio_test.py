# Test version without GPIO for FPS benchmarking

import os
import sys
import time
import threading
import cv2
import numpy as np
from urllib3.exceptions import InsecureRequestWarning
import requests
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from ultralytics import YOLO

# User-defined parameters
model_path = 'models/yolov8n-face_ncnn_model'
cam_source = 'https://192.168.1.111:8002/mjpeg'
resW, resH = 640, 480
min_thresh = 0.5

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit()

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize MJPEG stream with background thread
print(f'Initializing MJPEG stream: {cam_source}')
mjpeg_frame_buffer = {'frame': None, 'lock': threading.Lock()}
mjpeg_stop_event = threading.Event()

def mjpeg_reader_thread():
    """Background thread: read MJPEG stream and decode frames."""
    retry_count = 0
    while retry_count < 5 and not mjpeg_stop_event.is_set():
        try:
            session = requests.Session()
            response = session.get(cam_source, stream=True, verify=False, timeout=(2, 5))
            response.raise_for_status()
            print(f'[MJPEG] Connected, reading frames...')
            retry_count = 0
            mjpeg_buffer = bytes()
            
            for chunk in response.iter_content(chunk_size=8192):
                if mjpeg_stop_event.is_set():
                    break
                mjpeg_buffer += chunk
                
                a = mjpeg_buffer.find(b'\xff\xd8')
                b_end = mjpeg_buffer.find(b'\xff\xd9')
                
                if a != -1 and b_end != -1:
                    jpg = mjpeg_buffer[a:b_end+2]
                    mjpeg_buffer = mjpeg_buffer[b_end+2:]
                    decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        decoded = cv2.resize(decoded, (resW, resH))
                        with mjpeg_frame_buffer['lock']:
                            mjpeg_frame_buffer['frame'] = decoded
                    if len(mjpeg_buffer) > 1024*1024:
                        mjpeg_buffer = mjpeg_buffer[-512*1024:]
                        
        except Exception as e:
            retry_count += 1
            print(f'[MJPEG] Error: {e} (retry {retry_count}/5)')
            time.sleep(1)

reader = threading.Thread(target=mjpeg_reader_thread, daemon=True)
reader.start()
print('Connected to MJPEG stream (background thread active)')

# Wait for first frame
print('[MAIN] Waiting for first frame...')
frame_wait_start = time.time()
while time.time() - frame_wait_start < 10:
    with mjpeg_frame_buffer['lock']:
        if mjpeg_frame_buffer['frame'] is not None:
            print('[MAIN] First frame received!')
            break
    time.sleep(0.1)
else:
    print('[ERROR] No frames received in 10 seconds.')
    sys.exit(1)

# Initialize FPS tracking
frame_count = 0
fps_buffer = []
fps_avg_len = 60

print('[MAIN] Starting FPS benchmark... running for 30 seconds')
start_time = time.time()
end_time = start_time + 30  # 30 second benchmark

while time.time() < end_time:
    t_start = time.perf_counter()
    
    # Get latest frame
    with mjpeg_frame_buffer['lock']:
        frame = mjpeg_frame_buffer['frame']
    
    if frame is None:
        time.sleep(0.01)
        continue
    
    # Run inference
    results = model.track(frame, verbose=False)
    
    frame_count += 1
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate = 1.0 / (t_stop - t_start)
    
    if len(fps_buffer) >= fps_avg_len:
        fps_buffer.pop(0)
    fps_buffer.append(frame_rate)
    
    avg_fps = np.mean(fps_buffer)
    
    # Print every 10 frames
    if frame_count % 10 == 0:
        print(f'Frame {frame_count}: FPS = {avg_fps:.2f}')

# Show final results
elapsed = time.time() - start_time
final_fps = frame_count / elapsed
avg_fps = np.mean(fps_buffer) if fps_buffer else 0

print(f'\n========== BENCHMARK RESULTS ==========')
print(f'Frames processed: {frame_count}')
print(f'Elapsed time: {elapsed:.1f}s')
print(f'Average FPS: {final_fps:.2f}')
print(f'Instantaneous FPS (last {len(fps_buffer)} frames): {avg_fps:.2f}')
print(f'Model: {model_path}')
print(f'Resolution: {resW}x{resH}')
print(f'========================================')

# Cleanup
mjpeg_stop_event.set()
reader.join(timeout=2)
