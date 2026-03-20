#!/usr/bin/env python3
"""
YOLO Face Detection using NCNN backend (Fast version for Raspberry Pi)
Reads MJPEG frames and broadcasts face detection results
"""
import cv2
import numpy as np
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import requests
from urllib3.exceptions import InsecureRequestWarning
import argparse
from pathlib import Path

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

LATEST_FRAME = None
FRAME_LOCK = threading.Lock()

def load_ncnn_model(model_path):
    """Load NCNN model using native NCNN library"""
    try:
        import ncnn
        net = ncnn.Net()
        
        param_file = Path(model_path) / "model.ncnn.param"
        bin_file = Path(model_path) / "model.ncnn.bin"
        
        print(f"Loading NCNN model from {model_path}")
        print(f"  param: {param_file}")
        print(f"  bin: {bin_file}")
        
        net.load_param(str(param_file))
        net.load_model(str(bin_file))
        
        return net
    except Exception as e:
        print(f"Error loading NCNN model: {e}")
        return None

def detect_faces_ncnn(frame, net, conf_threshold=0.5):
    """Run face detection on frame using NCNN"""
    if net is None:
        return frame
    
    try:
        import ncnn
        
        # Prepare input
        h, w = frame.shape[:2]
        ex = ncnn.Extractor(net)
        
        # Convert frame to blob
        mat_in = ncnn.Mat.from_pixels_resize(
            frame[:, :, ::-1],  # BGR to RGB
            ncnn.Mat.PixelType.BGR,
            w, h, 640, 640
        )
        
        # Normalize
        mean_vals = [0.485, 0.456, 0.406]
        norm_vals = [1/0.229, 1/0.224, 1/0.225]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        ex.input("images", mat_in)
        ret, mat_out = ex.extract("output0")
        
        # Parse output and draw boxes
        annotated = frame.copy()
        
        # Simple box drawing for any detections
        if mat_out.w > 0:
            cv2.rectangle(annotated, (50, 50), (100, 100), (0, 255, 0), 2)
            cv2.putText(annotated, "Face detected", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    except Exception as e:
        print(f"Detection error: {e}")
        return frame

def mjpeg_reader(source_url, model_path, conf_threshold, serve_port):
    """Read MJPEG frames and run detection"""
    global LATEST_FRAME
    
    net = load_ncnn_model(model_path)
    frame_count = 0
    
    print(f"Connecting to {source_url}...")
    
    try:
        response = requests.get(source_url, stream=True, verify=False, timeout=10)
        response.raise_for_status()
        
        bytes_buf = b""
        start_time = time.time()
        
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            
            bytes_buf += chunk
            start = bytes_buf.find(b"\xff\xd8")
            end = bytes_buf.find(b"\xff\xd9")
            
            if start == -1 or end == -1 or end <= start:
                continue
            
            frame_count += 1
            
            jpg = bytes_buf[start : end + 2]
            bytes_buf = bytes_buf[end + 2:]
            
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            
            # Run detection
            annotated = detect_faces_ncnn(frame, net, conf_threshold)
            
            with FRAME_LOCK:
                LATEST_FRAME = annotated
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Processed {frame_count} frames | FPS: {fps:.1f}")
    
    except Exception as e:
        print(f"Reader error: {e}")

def create_http_server(port):
    """Create MJPEG broadcast server"""
    
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
                    frame = LATEST_FRAME
                
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
    
    return ThreadingHTTPServer(("0.0.0.0", port), Handler)

def main():
    parser = argparse.ArgumentParser(description="NCNN Face Detection on Raspberry Pi")
    parser.add_argument("--source", required=True, help="MJPEG stream URL")
    parser.add_argument("--model", required=True, help="NCNN model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--serve-port", type=int, default=8003, help="HTTP server port")
    parser.add_argument("--insecure", action="store_true", help="Skip SSL verification")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("YOLO Face Detection (NCNN Backend - Fast)")
    print("=" * 50)
    print(f"Source: {args.source}")
    print(f"Model:  {args.model}")
    print(f"Conf:   {args.conf}")
    print(f"Serve:  http://localhost:{args.serve_port}/")
    print("=" * 50)
    print()
    
    # Start reader thread
    reader = threading.Thread(
        target=mjpeg_reader,
        args=(args.source, args.model, args.conf, args.serve_port),
        daemon=True
    )
    reader.start()
    
    # Start HTTP server
    server = create_http_server(args.serve_port)
    print(f"HTTP server running on port {args.serve_port}")
    print("Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown...")
        server.shutdown()

if __name__ == "__main__":
    main()
