import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODINGS_PATH = BASE_DIR / "output" / "encodings.pkl"

# IP camera settings (MJPEG)
IP_URL = "http://192.168.1.24:8081/video"
IP_USERNAME = "admin"
IP_PASSWORD = "admin"

if IP_USERNAME and IP_PASSWORD:
    IP_URL = f"http://{IP_USERNAME}:{IP_PASSWORD}@192.168.1.24:8081/video"


def mjpeg_frames(stream_url: str):
    auth = (IP_USERNAME, IP_PASSWORD) if IP_USERNAME and IP_PASSWORD else None
    response = requests.get(stream_url, stream=True, timeout=5, auth=auth)
    response.raise_for_status()
    buffer = b""
    for chunk in response.iter_content(chunk_size=1024):
        buffer += chunk
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")
        if start != -1 and end != -1 and end > start:
            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is not None:
                yield frame


def check_stream_url(stream_url: str) -> None:
    try:
        auth = (IP_USERNAME, IP_PASSWORD) if IP_USERNAME and IP_PASSWORD else None
        response = requests.get(stream_url, stream=True, timeout=5, auth=auth)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "unknown")
        print(f"Stream check OK: {response.status_code} ({content_type})")
    except requests.RequestException as exc:
        print(f"Stream check failed: {exc}")


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(
            boolean_matches, loaded_encodings["names"], strict=False
        )
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    return None


def live_recognition(
    model: str = "hog",
    display: bool = True,
    save_frames: bool = False,
    save_every: int = 30,
) -> None:
    if not DEFAULT_ENCODINGS_PATH.exists():
        print("Error: Encodings file not found. Run training first.")
        return

    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    check_stream_url(IP_URL)

    cap = cv2.VideoCapture(IP_URL, cv2.CAP_FFMPEG)

    for _ in range(5):
        if cap.isOpened():
            break
        cap.release()
        cap = cv2.VideoCapture(IP_URL, cv2.CAP_FFMPEG)

    use_mjpeg_fallback = not cap.isOpened()
    if use_mjpeg_fallback:
        print("OpenCV could not open the stream. Falling back to MJPEG parser...")

    if use_mjpeg_fallback:
        frame_iter = mjpeg_frames(IP_URL)

        def next_frame():
            return True, next(frame_iter)

    else:

        def next_frame():
            return cap.read()

    frame_count = 0
    last_frame_path = BASE_DIR / "output" / "last_frame.jpg"

    while True:
        ret, frame = next_frame()
        if not ret:
            print("Error: Failed to read frame from stream.")
            break

        frame_count += 1

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb_small_frame, model=model
        )
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        for (top, right, bottom, left), encoding in zip(
            face_locations, face_encodings, strict=False
        ):
            name = _recognize_face(encoding, loaded_encodings) or "Unknown"

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(
                frame, (left, top), (right, bottom), (255, 0, 0), 2
            )
            cv2.rectangle(
                frame,
                (left, bottom - 20),
                (right, bottom),
                (255, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                name,
                (left + 5, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        if display:
            try:
                cv2.imshow("IP Camera Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                print(
                    "OpenCV GUI not available. Disabling display and saving frames to output/last_frame.jpg"
                )
                display = False
                save_frames = True

        if save_frames and (frame_count % max(save_every, 1) == 0):
            last_frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(last_frame_path), frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live face recognition with MJPEG fallback"
    )
    parser.add_argument(
        "-m",
        action="store",
        default="hog",
        choices=["hog", "cnn"],
        help="Which model to use: hog (CPU), cnn (GPU)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI display (useful on headless systems)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames to output/last_frame.jpg",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
        help="Save every Nth frame when saving is enabled",
    )
    args = parser.parse_args()

    live_recognition(
        model=args.m,
        display=not args.no_display,
        save_frames=args.save_frames or args.no_display,
        save_every=args.save_every,
    )
