#!/usr/bin/env python3
"""
YOLO Face Detection using yolodesk CLI wrapper
This is similar to EdjeElectronics yolo_detect.py but uses the yolodesk tool
which has all dependencies pre-installed.

Usage:
    python yolo_detect_face.py --model models/yolov8n-face.pt --source "https://192.168.1.111:8002/mjpeg" --conf 0.5 --no-display --serve --serve-port 8004
    python yolo_detect_face.py --model models/yolov8n-face.pt --source usb0 --conf 0.5
    python yolo_detect_face.py --model models/yolov8n-face.pt --source video.mp4 --conf 0.5
    python yolo_detect_face.py --model models/yolov8n-face.pt --source test_image.jpg --conf 0.5
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Face Detection using yolodesk CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO model file (e.g. models/yolov8n-face.pt)"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input source: usb0, camera index, image file, video file, or HTTP(S) MJPEG URL"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection (0-1)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow window (useful on headless systems)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Broadcast results via HTTP MJPEG stream"
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8004,
        help="Port for HTTP MJPEG broadcast"
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for HTTPS sources"
    )
    args = parser.parse_args()

    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Build yolodesk command
    cmd = [
        "python", "yolodesk",
        "--model", args.model,
        "--conf", str(args.conf),
        "live",
        "--source", args.source
    ]

    # Add optional flags
    if args.no_display:
        cmd.append("--no-display")
    if args.serve:
        cmd.extend(["--serve", "--serve-port", str(args.serve_port)])
    if args.insecure:
        cmd.append("--insecure")

    print(f"Running: {' '.join(cmd)}")
    print()

    # Execute yolodesk
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n✓ Detection stopped by user")
        sys.exit(0)
    except FileNotFoundError:
        print("ERROR: yolodesk not found. Make sure it's installed and in PATH.")
        sys.exit(1)


if __name__ == "__main__":
    main()
