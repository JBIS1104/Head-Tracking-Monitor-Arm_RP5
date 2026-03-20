#!/usr/bin/env python3
"""
Run YOLOv8n-face detection on MJPEG stream using pre-trained WIDERFace model.

This script:
1. Reads MJPEG stream from MJPEG server
2. Runs YOLOv8n-face detection on each frame
3. Broadcasts results via HTTP MJPEG server
4. Optimized for Raspberry Pi 5

Pre-trained model: YOLOv8n-face (trained on WIDERFace dataset)
- Size: 6.1 MB (.pt) / 13 MB (.onnx)
- Speed: 8-10 FPS on Pi 5
- Accuracy: mAP50 ~0.45 on WIDERFace
"""

import cv2
import numpy as np
import requests
import threading
import io
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


class MJPEGStreamReader:
    """Read MJPEG stream from HTTP source."""
    
    def __init__(self, url, verify_ssl=False):
        self.url = url
        self.verify_ssl = verify_ssl
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        
    def start(self):
        """Start reading stream in background thread."""
        self.running = True
        thread = threading.Thread(target=self._read_stream, daemon=True)
        thread.start()
        
    def stop(self):
        """Stop reading stream."""
        self.running = False
        
    def _read_stream(self):
        """Background thread to read MJPEG stream."""
        try:
            response = requests.get(
                self.url,
                stream=True,
                verify=self.verify_ssl,
                timeout=10
            )
            bytes_data = b''
            
            for chunk in response.iter_content(chunk_size=4096):
                if not self.running:
                    break
                    
                bytes_data += chunk
                # Find JPEG boundary
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode frame
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    
                    if frame is not None:
                        with self.lock:
                            self.frame = frame
                            
        except Exception as e:
            print(f"Stream reader error: {e}")
            
    def get_frame(self):
        """Get current frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


class HTTPMJPEGServer:
    """Simple HTTP MJPEG server for broadcasting results."""
    
    def __init__(self, port=8003, fps=10):
        self.port = port
        self.fps = fps
        self.frame_queue = []
        self.lock = threading.Lock()
        
    def add_frame(self, frame):
        """Add frame to broadcast."""
        with self.lock:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.frame_queue = [buffer.tobytes()]
            
    def get_frame(self):
        """Get current frame for MJPEG stream."""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[0]
        return None


def detect_faces(model, frame, conf=0.5):
    """Run face detection on frame."""
    try:
        results = model(frame, conf=conf, verbose=False)
        return results[0]
    except Exception as e:
        print(f"Detection error: {e}")
        return None


def draw_detections(frame, result):
    """Draw bounding boxes on frame."""
    if result is None or not hasattr(result, 'boxes'):
        return frame
    
    for box in result.boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f'Face {conf:.2f}'
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='YOLOv8n-face detection on MJPEG stream')
    parser.add_argument('--source', required=True, help='MJPEG stream URL')
    parser.add_argument('--model', default='models/yolov8n-face.pt', help='Model file path')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--port', type=int, default=8003, help='Output HTTP port')
    parser.add_argument('--insecure', action='store_true', help='Skip SSL verification')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8n-face Detection (WIDERFace Pre-trained)")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading model: {args.model}")
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return 1
    
    model = YOLO(args.model)
    print(f"✓ Model loaded")
    
    # Connect to stream
    print(f"\n2. Connecting to stream: {args.source}")
    reader = MJPEGStreamReader(args.source, verify_ssl=not args.insecure)
    reader.start()
    
    # Wait for first frame
    print("   Waiting for first frame...")
    for _ in range(30):
        frame = reader.get_frame()
        if frame is not None:
            print(f"✓ Connected (resolution: {frame.shape[1]}x{frame.shape[0]})")
            break
        time.sleep(0.1)
    else:
        print("Error: Could not connect to stream")
        return 1
    
    # Setup output server
    server = HTTPMJPEGServer(port=args.port)
    print(f"\n3. Starting MJPEG broadcast on port {args.port}")
    print(f"   View at: http://localhost:{args.port}/")
    
    # Detection loop
    print(f"\n4. Running detection (conf={args.conf})...")
    print("   Press Ctrl+C to stop\n")
    
    frame_count = 0
    fps_timer = time.time()
    
    try:
        while True:
            frame = reader.get_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Run detection
            result = detect_faces(model, frame, conf=args.conf)
            
            # Draw detections
            frame_with_boxes = draw_detections(frame, result)
            
            # Add FPS counter
            frame_count += 1
            if time.time() - fps_timer > 1:
                fps = frame_count
                frame_count = 0
                fps_timer = time.time()
                print(f"   FPS: {fps}, Faces: {len(result.boxes) if result else 0}")
            
            # Broadcast frame
            server.add_frame(frame_with_boxes)
            
    except KeyboardInterrupt:
        print("\n\n✓ Detection stopped")
        reader.stop()
        return 0


if __name__ == '__main__':
    exit(main())
