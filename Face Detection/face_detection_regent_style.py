# Bash:
# source "/home/billyp/Documents/Face Detection/.venv-gui/bin/activate"
# python "/home/billyp/Documents/Face Detection/face_detection_regent_style.py"


import argparse
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import requests


DEFAULT_URL = "http://192.168.1.24:8081/video"
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "admin"
DEFAULT_AUTH = "basic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime face detection from IP camera stream."
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=60,
        help="Minimum face size in pixels.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.1,
        help="Haar cascade scale factor.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Haar cascade minNeighbors.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable preview window.",
    )
    return parser.parse_args()


def with_basic_auth(url: str, user: Optional[str], password: Optional[str]) -> str:
    if not user or password is None:
        return url
    if "@" in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    return f"{scheme}://{user}:{password}@{rest}"


def open_stream(url: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    return None


def mjpeg_frames(
    url: str,
    user: Optional[str],
    password: Optional[str],
    auth_type: str,
) -> Generator[np.ndarray, None, None]:
    auth = None
    if auth_type != "none" and user and password is not None:
        if auth_type == "digest":
            auth = requests.auth.HTTPDigestAuth(user, password)
        else:
            auth = (user, password)

    with requests.get(url, stream=True, timeout=10, auth=auth) as response:
        response.raise_for_status()
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buffer.extend(chunk)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = buffer[start : end + 2]
                del buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame


def find_cascade() -> Optional[str]:
    cascade_name = "haarcascade_frontalface_default.xml"
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / cascade_name)
    candidates.extend(
        [
            Path(__file__).resolve().parent / "data" / cascade_name,
            Path("/usr/share/opencv4/haarcascades") / cascade_name,
            Path("/usr/share/opencv/haarcascades") / cascade_name,
            Path(cv2.__file__).resolve().parent / "data" / cascade_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def main() -> int:
    args = parse_args()

    cascade_path = find_cascade()
    if not cascade_path:
        print("Failed to locate Haar cascade file.")
        return 1

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Failed to load Haar cascade.")
        return 1

    url = with_basic_auth(DEFAULT_URL, DEFAULT_USER, DEFAULT_PASSWORD)
    cap = open_stream(url)
    use_mjpeg = cap is None
    if use_mjpeg:
        print(f"OpenCV failed to open stream, trying MJPEG parser: {DEFAULT_URL}")

    show_preview = not args.no_show
    min_size = (args.min_size, args.min_size)

    fps_count = 0
    fps_last = time.time()
    fps_value = 0.0

    frame_source = None
    if use_mjpeg:
        frame_source = mjpeg_frames(
            DEFAULT_URL, DEFAULT_USER, DEFAULT_PASSWORD, DEFAULT_AUTH
        )

    while True:
        if use_mjpeg:
            try:
                frame = next(frame_source)
                ok = True
            except StopIteration:
                ok = False
                frame = None
        else:
            ok, frame = cap.read()

        if not ok or frame is None:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.neighbors,
            minSize=min_size,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        fps_count += 1
        now = time.time()
        if now - fps_last >= 1.0:
            fps_value = fps_count / (now - fps_last)
            fps_count = 0
            fps_last = now

        if show_preview:
            cv2.putText(
                frame,
                f"Faces: {len(faces)}  FPS: {fps_value:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
