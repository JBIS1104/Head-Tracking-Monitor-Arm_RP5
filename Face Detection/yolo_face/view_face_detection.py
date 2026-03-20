#!/usr/bin/env python3
"""
Direct Python viewer for YOLOv8n-face detection.
Reads MJPEG stream, runs detection, displays results in GUI window.
No HTTP server overhead - just pure OpenCV display.

Usage:
    python view_face_detection.py --source https://192.168.1.111:8002/mjpeg --conf 0.6 --insecure
"""

import cv2
import argparse
import requests
import numpy as np
from pathlib import Path
import threading
import time
import sys

# Try to use ultralytics YOLO, fall back to OpenCV haar cascade
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("⚠️  ultralytics not available, using OpenCV Haar Cascade")


class MJPEGStreamReader:
    """Read MJPEG stream from HTTP source"""
    
    def __init__(self, url, verify_ssl=False):
        self.url = url
        self.verify_ssl = verify_ssl
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        
    def start(self):
        """Start reading stream in background thread"""
        self.running = True
        thread = threading.Thread(target=self._read_stream, daemon=True)
        thread.start()
        
    def stop(self):
        """Stop reading stream"""
        self.running = False
        
    def _read_stream(self):
        """Background thread to read MJPEG stream"""
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
                # Find JPEG boundary markers
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
            print(f"❌ Stream reader error: {e}")
            
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None


def detect_faces_haar(frame, cascade):
    """Detect faces using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    return faces


def detect_faces_yolo(model, frame, conf):
    """Detect faces using YOLO model"""
    try:
        results = model(frame, conf=conf, verbose=False)
        return results[0]
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
        return None


def draw_detections_haar(frame, faces):
    """Draw Haar Cascade detections"""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, 'Face',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )
    return frame


def draw_detections_yolo(frame, result):
    """Draw YOLO detections"""
    if result is None or not hasattr(result, 'boxes'):
        return frame
    
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Face {conf:.2f}'
        cv2.putText(
            frame, label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )
    return frame


def main():
    parser = argparse.ArgumentParser(
        description='Direct Python viewer for YOLOv8n-face detection'
    )
    parser.add_argument(
        '--source', required=True,
        help='MJPEG stream URL (e.g., https://192.168.1.111:8002/mjpeg)'
    )
    parser.add_argument(
        '--model', default='models/yolov8n-face.pt',
        help='Path to YOLO model file'
    )
    parser.add_argument(
        '--conf', type=float, default=0.6,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--insecure', action='store_true',
        help='Skip SSL certificate verification'
    )
    parser.add_argument(
        '--backend', choices=['auto', 'yolo', 'haar'], default='auto',
        help='Detection backend (auto=YOLO if available, else Haar)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8n-face Detection Viewer")
    print("=" * 60)
    print()
    
    # Determine backend
    use_yolo = (args.backend == 'yolo') or (args.backend == 'auto' and HAS_ULTRALYTICS)
    
    if use_yolo:
        print(f"Loading model: {args.model}")
        if not Path(args.model).exists():
            print(f"❌ Model not found at {args.model}")
            return 1
        model = YOLO(args.model)
        print("✓ YOLO model loaded")
    else:
        print("Loading Haar Cascade...")
        # Try multiple paths for Haar Cascade
        cascade_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/home/billyp/Documents/Face Detection/data/haarcascade_frontalface_default.xml',
            'data/haarcascade_frontalface_default.xml',
        ]
        cascade = None
        cascade_path = None
        for path in cascade_paths:
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascade = c
                cascade_path = path
                break
        
        if cascade is None or cascade.empty():
            print(f"❌ Failed to load Haar Cascade from any path")
            return 1
        print(f"✓ Haar Cascade loaded from {cascade_path}")
    
    # Connect to stream
    print(f"\nConnecting to stream: {args.source}")
    reader = MJPEGStreamReader(args.source, verify_ssl=not args.insecure)
    reader.start()
    
    # Wait for first frame
    print("Waiting for first frame...")
    for i in range(50):
        frame = reader.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            print(f"✓ Connected (resolution: {w}x{h})\n")
            break
        time.sleep(0.1)
    else:
        print("❌ Could not connect to stream")
        return 1
    
    # Main loop
    print(f"Detection running (backend={'YOLO' if use_yolo else 'Haar'}, conf={args.conf})")
    print("Press 'q' to quit, 's' to save frame\n")
    
    frame_count = 0
    fps_timer = time.time()
    face_count = 0
    
    try:
        while True:
            frame = reader.get_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Run detection
            if use_yolo:
                result = detect_faces_yolo(model, frame, args.conf)
                frame_with_boxes = draw_detections_yolo(frame, result)
                face_count = len(result.boxes) if result else 0
            else:
                faces = detect_faces_haar(frame, cascade)
                frame_with_boxes = draw_detections_haar(frame, faces)
                face_count = len(faces)
            
            # FPS counter
            frame_count += 1
            current_time = time.time()
            if current_time - fps_timer > 1:
                fps = frame_count
                frame_count = 0
                fps_timer = current_time
                print(f"FPS: {fps} | Faces: {face_count}")
            
            # Add FPS/face count to frame
            status = f"FPS: {fps if current_time - fps_timer > 0.5 else 'calculating...'} | Faces: {face_count}"
            cv2.putText(
                frame_with_boxes, status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            
            # Display frame
            cv2.imshow('Face Detection', frame_with_boxes)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Stopped")
                break
            elif key == ord('s'):
                filename = f'detection_frame_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame_with_boxes)
                print(f"✓ Saved: {filename}")
            
    except KeyboardInterrupt:
        print("\n✓ Stopped")
    finally:
        reader.stop()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
