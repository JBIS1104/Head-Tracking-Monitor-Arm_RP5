#!/usr/bin/env python3
"""
Optimized YOLO Face Detection using NCNN backend for Raspberry Pi 5.

Uses ncnn Python bindings directly (no torch/ultralytics required).
Targets ~6-8 FPS on Pi 5 with YOLOv8n-face NCNN model at 640x480.

Usage:
  USB camera:
    python face_detect_ncnn_optimized.py --model NCNN/yolov8n-face_ncnn_model --source usb0 --resolution 640x480

  MJPEG stream (headless, re-broadcast):
    python face_detect_ncnn_optimized.py --model NCNN/yolov8n-face_ncnn_model --source "https://192.168.1.100:8001/mjpeg" --no-display --serve --serve-port 8092

  Image file:
    python face_detect_ncnn_optimized.py --model NCNN/yolov8n-face_ncnn_model --source photo.jpg
"""

from __future__ import annotations

import argparse
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import ncnn
import numpy as np
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# ── Model ────────────────────────────────────────────────────────────────────

INPUT_SIZE = 640  # YOLOv8 expects 640x640


def load_model(model_dir: str) -> ncnn.Net:
    """Load NCNN model from directory."""
    model_path = Path(model_dir)
    param_file = model_path / "model.ncnn.param"
    bin_file = model_path / "model.ncnn.bin"

    if not param_file.exists() or not bin_file.exists():
        raise FileNotFoundError(
            f"NCNN model files not found in {model_path}. "
            f"Expected model.ncnn.param and model.ncnn.bin"
        )

    net = ncnn.Net()
    # Use all available cores on Pi 5 (4 cores)
    net.opt.num_threads = 4
    net.load_param(str(param_file))
    net.load_model(str(bin_file))
    print(f"Loaded NCNN model from {model_path}")
    return net


def detect_faces(
    net: ncnn.Net,
    frame: np.ndarray,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.45,
) -> list[dict]:
    """
    Run YOLOv8-face NCNN inference on a BGR frame.

    Returns list of detections:
        [{"bbox": (x1, y1, x2, y2), "conf": float, "keypoints": [(x,y,v), ...]}, ...]
    """
    h, w = frame.shape[:2]

    # Letterbox: scale to 640x640 preserving aspect ratio
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w, pad_h = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2

    # Use ncnn Mat for efficient pixel conversion + resize
    mat_in = ncnn.Mat.from_pixels_resize(
        frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, new_w, new_h
    )

    # Pad to 640x640
    mat_padded = ncnn.copy_make_border(
        mat_in,
        pad_h, INPUT_SIZE - new_h - pad_h,
        pad_w, INPUT_SIZE - new_w - pad_w,
        ncnn.BorderType.BORDER_CONSTANT, 114.0,
    )

    # Normalize to [0, 1]
    mat_padded.substract_mean_normalize([0, 0, 0], [1 / 255.0, 1 / 255.0, 1 / 255.0])

    # Inference
    ex = net.create_extractor()
    ex.input("in0", mat_padded)
    _, out = ex.extract("out0")

    # Output shape: (20, 8400)
    #   rows 0-3: cx, cy, w, h (in 640x640 space)
    #   row 4: face confidence
    #   rows 5-19: 5 keypoints * (x, y, visibility)
    output = np.array(out)  # (20, 8400)

    # Filter by confidence
    scores = output[4, :]
    mask = scores > conf_threshold
    if not mask.any():
        return []

    filtered = output[:, mask]
    scores = filtered[4, :]

    # Convert cx,cy,w,h to x1,y1,x2,y2
    cx, cy, bw, bh = filtered[0], filtered[1], filtered[2], filtered[3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # NMS
    boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    detections = []
    for idx in indices:
        i = idx if isinstance(idx, int) else idx[0]

        # Map back from letterboxed 640x640 to original frame coords
        det_x1 = (x1[i] - pad_w) / scale
        det_y1 = (y1[i] - pad_h) / scale
        det_x2 = (x2[i] - pad_w) / scale
        det_y2 = (y2[i] - pad_h) / scale

        # Clamp to frame boundaries
        det_x1 = max(0, min(w, det_x1))
        det_y1 = max(0, min(h, det_y1))
        det_x2 = max(0, min(w, det_x2))
        det_y2 = max(0, min(h, det_y2))

        # Keypoints (5 face landmarks)
        keypoints = []
        for k in range(5):
            kx = (filtered[5 + k * 3, i] - pad_w) / scale
            ky = (filtered[6 + k * 3, i] - pad_h) / scale
            kv = filtered[7 + k * 3, i]
            keypoints.append((kx, ky, kv))

        detections.append({
            "bbox": (int(det_x1), int(det_y1), int(det_x2), int(det_y2)),
            "conf": float(scores[i]),
            "keypoints": keypoints,
        })

    return detections


# ── Drawing ──────────────────────────────────────────────────────────────────

GREEN = (0, 255, 0)
CYAN = (255, 255, 0)


def draw_detections(frame: np.ndarray, detections: list[dict], fps: float = 0) -> np.ndarray:
    """Draw bounding boxes, keypoints, and FPS on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
        label = f"face {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), GREEN, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Keypoints
        for kx, ky, kv in det["keypoints"]:
            if kv > 0.5:
                cv2.circle(annotated, (int(kx), int(ky)), 3, CYAN, -1)

    # FPS counter
    if fps > 0:
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

    return annotated


# ── MJPEG Broadcaster ───────────────────────────────────────────────────────

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
                        frame_data = broadcaster._last_jpeg
                    if frame_data is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_data)}\r\n\r\n".encode())
                        self.wfile.write(frame_data)
                        self.wfile.write(b"\r\n")
                    except Exception:
                        break

            def log_message(self, format, *args):
                return

        return _ThreadingHTTPServer((self.host, self.port), Handler)

    def start(self):
        self._thread.start()

    def update(self, bgr_frame: np.ndarray):
        ok, enc = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._condition:
            self._last_jpeg = enc.tobytes()
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ── Source Helpers ───────────────────────────────────────────────────────────

def parse_source(source: str):
    s = source.strip()
    if s.lower().startswith("usb") and s[3:].isdigit():
        return int(s[3:])
    if s.isdigit():
        return int(s)
    return s


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    if "x" not in value.lower():
        raise ValueError("resolution must be WxH, e.g. 640x480")
    w, h = value.lower().split("x", 1)
    return int(w), int(h)


def highgui_available() -> bool:
    try:
        cv2.namedWindow("_test_")
        cv2.destroyWindow("_test_")
        return True
    except Exception:
        return False


# ── Run Loops ────────────────────────────────────────────────────────────────

def run_camera(
    net: ncnn.Net,
    source: int,
    size: tuple[int, int] | None,
    conf: float,
    display: bool,
    broadcaster: MjpegBroadcaster | None,
):
    """Run detection on a USB camera."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {source}")
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    fps_counter = FpsCounter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break

            detections = detect_faces(net, frame, conf)
            fps = fps_counter.tick()
            annotated = draw_detections(frame, detections, fps)

            if display:
                cv2.imshow("Face Detection (NCNN)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if broadcaster:
                broadcaster.update(annotated)
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def run_mjpeg_stream(
    net: ncnn.Net,
    url: str,
    conf: float,
    display: bool,
    insecure: bool,
    broadcaster: MjpegBroadcaster | None,
):
    """Run detection on an MJPEG stream URL."""
    print(f"Connecting to {url} ...")
    with requests.get(url, stream=True, verify=not insecure, timeout=10) as resp:
        resp.raise_for_status()
        buf = b""
        fps_counter = FpsCounter()

        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buf += chunk

            # Find JPEG boundaries
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9")
            if start == -1 or end == -1 or end <= start:
                continue

            jpg = buf[start:end + 2]
            buf = buf[end + 2:]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            detections = detect_faces(net, frame, conf)
            fps = fps_counter.tick()
            annotated = draw_detections(frame, detections, fps)

            if display:
                cv2.imshow("Face Detection (NCNN)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if broadcaster:
                broadcaster.update(annotated)

    if display:
        cv2.destroyAllWindows()


def run_image(net: ncnn.Net, path: str, conf: float):
    """Run detection on a single image."""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    detections = detect_faces(net, frame, conf)
    annotated = draw_detections(frame, detections)

    print(f"Detected {len(detections)} face(s)")
    for i, det in enumerate(detections):
        print(f"  #{i+1}: bbox={det['bbox']} conf={det['conf']:.3f}")

    cv2.imshow("Face Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── FPS Counter ──────────────────────────────────────────────────────────────

class FpsCounter:
    def __init__(self, window: int = 30):
        self._window = window
        self._times: list[float] = []
        self._fps = 0.0

    def tick(self) -> float:
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            if elapsed > 0:
                self._fps = (len(self._times) - 1) / elapsed
        return self._fps


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimized YOLO Face Detection (NCNN) for Raspberry Pi 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="NCNN model directory")
    parser.add_argument("--source", default="usb0", help="usb0, camera index, image path, or MJPEG URL")
    parser.add_argument("--resolution", default="640x480", help="Camera resolution WxH")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window")
    parser.add_argument("--serve", action="store_true", help="Broadcast results as MJPEG")
    parser.add_argument("--serve-host", default="0.0.0.0", help="Broadcast host")
    parser.add_argument("--serve-port", type=int, default=8092, help="Broadcast port")
    args = parser.parse_args()

    net = load_model(args.model)
    display = not args.no_display
    if display and not highgui_available():
        print("No display available, switching to headless mode.")
        display = False

    broadcaster = None
    if args.serve:
        broadcaster = MjpegBroadcaster(host=args.serve_host, port=args.serve_port)
        broadcaster.start()
        print(f"MJPEG broadcast: http://{args.serve_host}:{args.serve_port}/")

    try:
        src = parse_source(args.source)
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            run_mjpeg_stream(net, src, args.conf, display, args.insecure, broadcaster)
        elif isinstance(src, int):
            size = parse_resolution(args.resolution)
            run_camera(net, src, size, args.conf, display, broadcaster)
        else:
            # File path (image or video)
            p = Path(src)
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                run_image(net, src, args.conf)
            else:
                # Treat as video file - use camera loop with file path
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open: {src}")
                fps_counter = FpsCounter()
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        detections = detect_faces(net, frame, args.conf)
                        fps = fps_counter.tick()
                        annotated = draw_detections(frame, detections, fps)
                        if display:
                            cv2.imshow("Face Detection (NCNN)", annotated)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        if broadcaster:
                            broadcaster.update(annotated)
                finally:
                    cap.release()
                    if display:
                        cv2.destroyAllWindows()
    finally:
        if broadcaster:
            broadcaster.stop()


if __name__ == "__main__":
    main()
