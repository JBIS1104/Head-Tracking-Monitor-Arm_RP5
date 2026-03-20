# Performance profiling version to identify bottlenecks

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

model_path = 'models/yolov8n-face_ncnn_model'
cam_source = 'https://192.168.1.111:8002/mjpeg'
resW, resH = 640, 480

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit()

print('[PROFILE] Loading model...')
t0 = time.time()
model = YOLO(model_path, task='detect')
print(f'[PROFILE] Model load time: {time.time()-t0:.3f}s')

# MJPEG reader
mjpeg_frame_buffer = {'frame': None, 'lock': threading.Lock()}
mjpeg_stop_event = threading.Event()

def mjpeg_reader_thread():
    retry_count = 0
    while retry_count < 5 and not mjpeg_stop_event.is_set():
        try:
            session = requests.Session()
            response = session.get(cam_source, stream=True, verify=False, timeout=(2, 5))
            response.raise_for_status()
            print(f'[MJPEG] Connected')
            retry_count = 0
            mjpeg_buffer = b''
            last_end_pos = 0
            
            for chunk in response.iter_content(chunk_size=32768):
                if mjpeg_stop_event.is_set():
                    break
                mjpeg_buffer += chunk
                
                while True:
                    jpg_start = mjpeg_buffer.find(b'\xff\xd8', last_end_pos)
                    if jpg_start == -1:
                        if len(mjpeg_buffer) > 512*1024:
                            mjpeg_buffer = mjpeg_buffer[-256*1024:]
                            last_end_pos = 0
                        break
                    
                    jpg_end = mjpeg_buffer.find(b'\xff\xd9', jpg_start + 2)
                    if jpg_end == -1:
                        break
                    
                    jpg = mjpeg_buffer[jpg_start:jpg_end+2]
                    last_end_pos = jpg_end + 2
                    
                    decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None and decoded.size > 0:
                        h, w = decoded.shape[:2]
                        if (h, w) != (resH, resW):
                            decoded = cv2.resize(decoded, (resW, resH), interpolation=cv2.INTER_LINEAR)
                        with mjpeg_frame_buffer['lock']:
                            mjpeg_frame_buffer['frame'] = decoded
                        
        except Exception as e:
            retry_count += 1
            print(f'[MJPEG] Error: {e}')
            time.sleep(1)

reader = threading.Thread(target=mjpeg_reader_thread, daemon=True)
reader.start()

# Wait for first frame
print('[MAIN] Waiting for first frame...')
for i in range(100):
    with mjpeg_frame_buffer['lock']:
        if mjpeg_frame_buffer['frame'] is not None:
            print('[MAIN] First frame received!')
            break
    time.sleep(0.1)
else:
    print('[ERROR] No frames received')
    sys.exit(1)

# Timing profile
print('[PROFILE] Starting 30s benchmark with timing breakdown...')
print('=' * 70)

frame_count = 0
start_time = time.time()
end_time = start_time + 30

timing = {
    'frame_acquisition': [],
    'model_inference': [],
    'total': [],
}

while time.time() < end_time:
    loop_start = time.perf_counter()
    
    # Get frame
    acq_start = time.perf_counter()
    with mjpeg_frame_buffer['lock']:
        frame = mjpeg_frame_buffer['frame']
    acq_time = time.perf_counter() - acq_start
    
    if frame is None:
        time.sleep(0.01)
        continue
    
    # NCNN inference
    inf_start = time.perf_counter()
    results = model.track(frame, verbose=False)
    inf_time = time.perf_counter() - inf_start
    
    total_time = time.perf_counter() - loop_start
    
    timing['frame_acquisition'].append(acq_time)
    timing['model_inference'].append(inf_time)
    timing['total'].append(total_time)
    
    frame_count += 1
    
    if frame_count % 5 == 0:
        avg_fps = frame_count / (time.time() - start_time)
        avg_acq = np.mean(timing['frame_acquisition'][-5:]) * 1000
        avg_inf = np.mean(timing['model_inference'][-5:]) * 1000
        print(f"Frame {frame_count:3d}: FPS={avg_fps:.2f} | Acq={avg_acq:.1f}ms | Infer={avg_inf:.1f}ms")

elapsed = time.time() - start_time
final_fps = frame_count / elapsed

print('=' * 70)
print(f'[RESULTS] Total frames: {frame_count}')
print(f'[RESULTS] Elapsed time: {elapsed:.1f}s')
print(f'[RESULTS] Average FPS: {final_fps:.2f}')
print(f'[RESULTS] Frame acquisition:  {np.mean(timing["frame_acquisition"])*1000:.2f}ms avg')
print(f'[RESULTS] Model inference:     {np.mean(timing["model_inference"])*1000:.2f}ms avg')
print(f'[RESULTS] Total per-frame:     {np.mean(timing["total"])*1000:.2f}ms avg')
print('=' * 70)

mjpeg_stop_event.set()
reader.join(timeout=2)
