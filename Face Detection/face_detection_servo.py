# Bash:
# source "/home/billyp/Documents/Face Detection/.venv-gui/bin/activate"
# python "/home/billyp/Documents/Face Detection/face_detection_servo.py" --insecure

import argparse
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import requests
import urllib3
from gpiozero import PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

DEFAULT_URL = "https://192.168.1.111:8001/mjpeg"
DEFAULT_USER: Optional[str] = None
DEFAULT_PASSWORD: Optional[str] = None
DEFAULT_AUTH = "none"

PWM_PIN = 18
PWM_PIN_2 = 13
PWM_FREQUENCY = 400
SERVO_STEP = 0.005  # 0.5%
SERVO_INTERVAL = 0.1  # 100 ms
SERVO_MIN = 0.0
SERVO_MAX = 1.0
SERVO_NEUTRAL = 0.5
SERVO_DEADZONE_RATIO = 0.15
SERVO_VERTICAL_HOME = 0.30
SERVO_VERTICAL_START = 0.65
SERVO_RAMP_STEP = 0.003
SERVO_RAMP_INTERVAL = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime face detection with servo tracking."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Camera or MJPEG stream URL.",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
        help="Username for stream authentication.",
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help="Password for stream authentication.",
    )
    parser.add_argument(
        "--auth",
        choices=["none", "basic", "digest"],
        default=DEFAULT_AUTH,
        help="Authentication type for MJPEG parser.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for HTTPS streams.",
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
    verify_tls: bool,
    reconnect_delay: float = 2.0,
) -> Generator[np.ndarray, None, None]:
    auth = None
    if auth_type != "none" and user and password is not None:
        if auth_type == "digest":
            auth = requests.auth.HTTPDigestAuth(user, password)
        else:
            auth = (user, password)

    tls_hint_shown = False
    current_verify_tls = verify_tls

    while True:
        try:
            with requests.get(
                url,
                stream=True,
                timeout=10,
                auth=auth,
                verify=current_verify_tls,
            ) as response:
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
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            yield frame
        except requests.exceptions.SSLError as exc:
            print(
                f"MJPEG TLS error: {exc}. Retrying in {reconnect_delay:.1f}s..."
            )
            if current_verify_tls and not tls_hint_shown:
                print(
                    "Hint: self-signed certificate detected. "
                    "Falling back to insecure TLS mode for this session. "
                    "Use --insecure to skip this check at startup."
                )
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                tls_hint_shown = True
                current_verify_tls = False
            time.sleep(reconnect_delay)
        except requests.RequestException as exc:
            print(
                f"MJPEG connection error: {exc}. Retrying in {reconnect_delay:.1f}s..."
            )
            time.sleep(reconnect_delay)


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


def pick_primary_face(faces: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def update_servo(duty: float, center_pos: float, axis_size: int) -> float:
    mid = axis_size / 2.0
    deadzone_half = axis_size * SERVO_DEADZONE_RATIO / 2.0
    if center_pos < mid - deadzone_half:
        duty += SERVO_STEP
    elif center_pos > mid + deadzone_half:
        duty -= SERVO_STEP
    return max(SERVO_MIN, min(SERVO_MAX, duty))


def move_servo_smoothly(
    pwm: PWMOutputDevice,
    start_duty: float,
    target_duty: float,
    step: float = SERVO_RAMP_STEP,
    interval: float = SERVO_RAMP_INTERVAL,
) -> float:
    current = max(SERVO_MIN, min(SERVO_MAX, start_duty))
    target = max(SERVO_MIN, min(SERVO_MAX, target_duty))
    pwm.value = current

    if abs(target - current) < 1e-9:
        return target

    direction = 1.0 if target > current else -1.0
    while (direction > 0 and current < target) or (direction < 0 and current > target):
        current += direction * step
        if direction > 0 and current > target:
            current = target
        elif direction < 0 and current < target:
            current = target
        pwm.value = current
        time.sleep(interval)

    return target


def main() -> int:
    args = parse_args()

    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    cascade_path = find_cascade()
    if not cascade_path:
        print("Failed to locate Haar cascade file.")
        return 1

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Failed to load Haar cascade.")
        return 1

    url = with_basic_auth(args.url, args.user, args.password)
    lower_url = args.url.lower()
    prefer_mjpeg_parser = lower_url.startswith(("http://", "https://")) and (
        "/mjpeg" in lower_url
    )

    if prefer_mjpeg_parser:
        cap = None
        use_mjpeg = True
        print(f"Using MJPEG parser for stream: {args.url}")
    else:
        cap = open_stream(url)
        use_mjpeg = cap is None
        if use_mjpeg:
            print(f"OpenCV failed to open stream, trying MJPEG parser: {args.url}")

    show_preview = not args.no_show
    min_size = (args.min_size, args.min_size)

    factory = LGPIOFactory()
    pwm_x = PWMOutputDevice(PWM_PIN, frequency=PWM_FREQUENCY, pin_factory=factory)
    pwm_y = PWMOutputDevice(
        PWM_PIN_2, frequency=PWM_FREQUENCY, pin_factory=factory
    )
    duty_x = SERVO_NEUTRAL
    duty_y = SERVO_VERTICAL_HOME
    pwm_x.value = duty_x
    pwm_y.value = duty_y
    duty_y = move_servo_smoothly(pwm_y, duty_y, SERVO_VERTICAL_START)
    last_servo_update = time.time()

    fps_count = 0
    fps_last = time.time()
    fps_value = 0.0

    frame_source = None
    if use_mjpeg:
        frame_source = mjpeg_frames(
            args.url,
            args.user,
            args.password,
            args.auth,
            verify_tls=not args.insecure,
        )

    safe_stop_requested = False

    try:
        while True:
            if use_mjpeg:
                frame = next(frame_source)
                ok = frame is not None
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

            primary = pick_primary_face(faces)
            if primary is not None:
                x, y, w, h = primary
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                now = time.time()
                if now - last_servo_update >= SERVO_INTERVAL:
                    duty_x = update_servo(duty_x, x + w / 2.0, frame.shape[1])
                    duty_y = update_servo(duty_y, y + h / 2.0, frame.shape[0])
                    pwm_x.value = duty_x
                    pwm_y.value = duty_y
                    last_servo_update = now

            fps_count += 1
            now = time.time()
            if now - fps_last >= 1.0:
                fps_value = fps_count / (now - fps_last)
                fps_count = 0
                fps_last = now

            if show_preview:
                cv2.putText(
                    frame,
                    f"Faces: {len(faces)}  FPS: {fps_value:.1f}  X: {duty_x*100:.1f}%  Y: {duty_y*100:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Face Detection Servo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    safe_stop_requested = True
                    break
    except KeyboardInterrupt:
        safe_stop_requested = True
    finally:
        if safe_stop_requested:
            duty_x = move_servo_smoothly(pwm_x, duty_x, SERVO_NEUTRAL)
            duty_y = move_servo_smoothly(pwm_y, duty_y, SERVO_VERTICAL_HOME)
        if cap is not None:
            cap.release()
        pwm_x.close()
        pwm_y.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
