import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import requests
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODINGS_PATH = BASE_DIR / "output" / "encodings.pkl"
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
(BASE_DIR / "training").mkdir(exist_ok=True)
(BASE_DIR / "output").mkdir(exist_ok=True)
(BASE_DIR / "validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "--live", action="store_true", help="Run live recognition from IP camera"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

# IP camera settings (MJPEG)
IP_URL = "http://192.168.1.24:8081/video"
IP_USERNAME = "admin"
IP_PASSWORD = "admin"

if IP_USERNAME and IP_PASSWORD:
    IP_URL = f"http://{IP_USERNAME}:{IP_PASSWORD}@192.168.1.24:8081/video"


def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = [
        p
        for p in (BASE_DIR / "training").rglob("*")
        if p.is_file() and p.suffix.lower() in supported_exts
    ]

    if not image_paths:
        print("No training images found under ./training")
        return

    print(f"Found {len(image_paths)} training images. Encoding faces...")

    for idx, filepath in enumerate(image_paths, start=1):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        if not face_locations:
            if idx % 10 == 0 or idx == len(image_paths):
                print(f"Processed {idx}/{len(image_paths)} images (no face)")
            continue
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} images")

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    print(f"Saved {len(encodings)} encodings to {encodings_location}")


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings, strict=False
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


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


def live_recognition(model: str = "hog"):
    """
    Run live face recognition from an IP camera stream.
    """
    if not DEFAULT_ENCODINGS_PATH.exists():
        print("Error: Encodings file not found. Run --train first.")
        return

    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

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

    while True:
        ret, frame = next_frame()
        if not ret:
            print("Error: Failed to read frame from stream.")
            break

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

        cv2.imshow("IP Camera Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
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


def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    for filepath in (BASE_DIR / "validation").rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in supported_exts:
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.live:
        live_recognition(model=args.m)
