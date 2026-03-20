#!/usr/bin/env python3
"""
Raspberry Pi YOLO Face Detection runner (NCNN optimized for Pi 5)
Based on: https://www.ejtech.io/learn/yolo-on-raspberry-pi
Uses: YOLOv8n-Face from https://github.com/lindevs/yolov8-face

Expected setup flow:
1) python3 -m venv --system-site-packages venv
2) source venv/bin/activate
3) pip install ultralytics ncnn opencv-python
4) Download model: wget https://github.com/lindevs/yolov8-face/releases/download/v1.0.1/yolov8n-face-lindevs.pt
5) Export to NCNN: yolo export model=yolov8n-face-lindevs.pt format=ncnn
6) python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --resolution=640x480

Run detection:
  - USB camera: python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=usb0 --resolution=640x480
  - Image: python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=test.jpg
  - Video: python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source=test.mp4
  - MJPEG stream: python yolo_face_detect_rpi.py --model=yolov8n-face-lindevs_ncnn_model --source="http://192.168.1.100:8080/video" --no-display --serve --serve-port=8092
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import requests


def parse_source(source: str):
    """Parse source string to camera index or path."""
    s = source.strip()
    if s.lower().startswith("usb") and s[3:].isdigit():
        return int(s[3:])
    if s.isdigit():
        return int(s)
    return s


def highgui_available() -> bool:
    """Check if OpenCV HighGUI is available."""
    try:
        cv2.namedWindow("_cv_test_")
        cv2.destroyWindow("_cv_test_")
        return True
    except Exception:
        return False


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MjpegBroadcaster:
    """Broadcast annotated frames as MJPEG stream over HTTP."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8092):
        self.host = host
        self.port = port
        self._last_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stopped = False
        self._server = self._build_server()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _build_server(self):
        broadcaster = self

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
                    with broadcaster._condition:
                        broadcaster._condition.wait_for(
                            lambda: broadcaster._last_jpeg is not None or broadcaster._stopped
                        )
                        if broadcaster._stopped:
                            break
                        frame = broadcaster._last_jpeg

                    if frame is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except Exception:
                        break

            def log_message(self, format, *args):
                return

        return _ThreadingHTTPServer((self.host, self.port), Handler)

    def start(self) -> None:
        """Start the MJPEG broadcast server."""
        self._thread.start()

    def update(self, bgr_frame) -> None:
        """Update with a new frame to broadcast."""
        ok, enc = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._condition:
            self._last_jpeg = enc.tobytes()
            self._condition.notify_all()

    def stop(self) -> None:
        """Stop the MJPEG broadcast server."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    """Parse WxH resolution string."""
    if not value:
        return None
    if "x" not in value.lower():
        raise ValueError("resolution must be in WxH format, e.g. 640x480")
    w_str, h_str = value.lower().split("x", 1)
    w, h = int(w_str), int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError("resolution width/height must be > 0")
    return (w, h)


def open_capture(src, size: tuple[int, int] | None):
    """Open video capture from camera or file."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to receive frames from source: {src}")
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def run_live(
    model,
    source,
    size: tuple[int, int] | None,
    conf: float,
    display: bool,
    insecure: bool,
    broadcaster: MjpegBroadcaster | None,
) -> None:
    """Run face detection on live camera feed or MJPEG stream."""
    if display and not highgui_available():
        print("OpenCV HighGUI unavailable; switching to --no-display behavior.")
        display = False

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        run_live_mjpeg_url(
            model=model,
            url=source,
            conf=conf,
            display=display,
            insecure=insecure,
            broadcaster=broadcaster,
        )
        return

    cap = open_capture(source, size)
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Unable to receive frames from the camera/source.")
                break

            result = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = result.plot()
            
            frame_count += 1
            if frame_count % 30 == 0:
                num_faces = len(result.boxes) if hasattr(result, 'boxes') else 0
                print(f"Frame {frame_count}: {num_faces} face(s) detected")

            if display:
                cv2.imshow("YOLO Face Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if broadcaster is not None:
                broadcaster.update(annotated)
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def run_live_mjpeg_url(
    model,
    url: str,
    conf: float,
    display: bool,
    insecure: bool,
    broadcaster: MjpegBroadcaster | None,
) -> None:
    """Run face detection on MJPEG stream from URL."""
    with requests.get(url, stream=True, verify=not insecure, timeout=10) as response:
        response.raise_for_status()
        bytes_buf = b""
        frame_count = 0

        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            bytes_buf += chunk
            start = bytes_buf.find(b"\xff\xd8")
            end = bytes_buf.find(b"\xff\xd9")
            if start == -1 or end == -1 or end <= start:
                continue

            jpg = bytes_buf[start : end + 2]
            bytes_buf = bytes_buf[end + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result = model.predict(frame, conf=conf, verbose=False)[0]
            annotated = result.plot()
            
            frame_count += 1
            if frame_count % 30 == 0:
                num_faces = len(result.boxes) if hasattr(result, 'boxes') else 0
                print(f"Frame {frame_count}: {num_faces} face(s) detected")

            if display:
                cv2.imshow("YOLO Face Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if broadcaster is not None:
                broadcaster.update(annotated)

    if display:
        cv2.destroyAllWindows()


def run_file(model, source, conf: float) -> None:
    """Run face detection on image or video file."""
    model.predict(source=source, conf=conf, show=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO Face Detection NCNN model on Raspberry Pi (image/video/camera)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="NCNN model folder path (e.g. yolov8n-face-lindevs_ncnn_model)",
    )
    parser.add_argument(
        "--source",
        default="usb0",
        help="Input source: usb0, camera index (0), image, video, folder, or HTTP(S) MJPEG URL",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Optional display resolution in WxH, e.g. 640x480 (camera mode)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection (0-1)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for HTTPS MJPEG sources",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow window (useful on headless Pi)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve processed frames at /stream.mjpeg",
    )
    parser.add_argument(
        "--serve-host",
        default="0.0.0.0",
        help="Host for processed MJPEG server",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8092,
        help="Port for processed MJPEG server",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is required for this script. Install with: pip install ultralytics ncnn"
        ) from exc

    model = YOLO(str(model_path))
    broadcaster: MjpegBroadcaster | None = None
    if args.serve:
        broadcaster = MjpegBroadcaster(host=args.serve_host, port=args.serve_port)
        broadcaster.start()
        print(f"Processed stream at: http://{args.serve_host}:{args.serve_port}/")

    try:
        src = parse_source(args.source)
        if isinstance(src, int) or (isinstance(src, str) and src.lower().startswith("usb")) or (
            isinstance(src, str) and src.startswith(("http://", "https://"))
        ):
            size = parse_resolution(args.resolution)
            run_live(
                model=model,
                source=parse_source(args.source),
                size=size,
                conf=args.conf,
                display=not args.no_display,
                insecure=args.insecure,
                broadcaster=broadcaster,
            )
        else:
            run_file(model=model, source=src, conf=args.conf)
    finally:
        if broadcaster is not None:
            broadcaster.stop()


if __name__ == "__main__":
    main()
