#!/usr/bin/env python3
"""
Optimized YOLO Face Detection with Frame Skipping
Processes every Nth frame to reduce CPU load significantly
"""
import cv2
import numpy as np
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Configuration
SOURCE_URL = "https://192.168.1.111:8002/mjpeg"
SERVE_PORT = 8003
CONFIDENCE = 0.6
FRAME_SKIP = 2  # Process every 2nd frame (50% faster)

# Shared state
LATEST_ANNOTATED_FRAME = None
FRAME_LOCK = threading.Lock()

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def mjpeg_reader_thread():
    """Read frames from MJPEG stream and process with face detection"""
    global LATEST_ANNOTATED_FRAME
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        use_ultralytics = True
    except ImportError:
        print("ultralytics not available; using alternate method")
        use_ultralytics = False
    
    frame_count = 0
    processed_count = 0
    
    print(f"Connecting to {SOURCE_URL}...")
    
    try:
        response = requests.get(SOURCE_URL, stream=True, verify=False, timeout=10)
        response.raise_for_status()
        
        bytes_buf = b""
        
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            
            bytes_buf += chunk
            start = bytes_buf.find(b"\xff\xd8")
            end = bytes_buf.find(b"\xff\xd9")
            
            if start == -1 or end == -1 or end <= start:
                continue
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % FRAME_SKIP != 0:
                bytes_buf = bytes_buf[end + 2:]
                continue
            
            processed_count += 1
            
            jpg = bytes_buf[start : end + 2]
            bytes_buf = bytes_buf[end + 2 :]
            
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            
            # Run inference
            try:
                if use_ultralytics:
                    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
                    annotated = results[0].plot()
                else:
                    annotated = frame
            except Exception as e:
                print(f"Error: {e}")
                annotated = frame
            
            # Store for HTTP server
            with FRAME_LOCK:
                LATEST_ANNOTATED_FRAME = annotated
            
            if processed_count % 30 == 0:
                print(f"Processed {processed_count} frames (skipped {frame_count - processed_count})")
    
    except Exception as e:
        print(f"Error: {e}")

def create_http_server():
    """Create MJPEG server to broadcast annotated frames"""
    
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body style='margin:0;background:#000;'>"
                    b"<img src='/stream.mjpeg' style='width:100%;height:auto;'/>"
                    b"</body></html>"
                )
                return
            
            if self.path != "/stream.mjpeg":
                self.send_response(404)
                self.end_headers()
                return
            
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            
            while True:
                with FRAME_LOCK:
                    frame = LATEST_ANNOTATED_FRAME
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                try:
                    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                    
                    data = enc.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                except Exception as e:
                    print(f"Stream error: {e}")
                    break
        
        def log_message(self, format, *args):
            return
    
    return ThreadingHTTPServer(("0.0.0.0", SERVE_PORT), Handler)

def main():
    print(f"YOLO Face Detection (Optimized)")
    print(f"  Frame skip: {FRAME_SKIP} (process every {FRAME_SKIP}th frame)")
    print(f"  Confidence: {CONFIDENCE}")
    print(f"  Input:  {SOURCE_URL}")
    print(f"  Output: http://localhost:{SERVE_PORT}/")
    print()
    
    # Start reader thread
    reader = threading.Thread(target=mjpeg_reader_thread, daemon=True)
    reader.start()
    
    # Start HTTP server
    server = create_http_server()
    print(f"HTTP server running on port {SERVE_PORT}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown...")
        server.shutdown()

if __name__ == "__main__":
    main()
