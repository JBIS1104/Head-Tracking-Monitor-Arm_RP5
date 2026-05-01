#!/usr/bin/env python3
"""
monitor_arm_track.py — YOLO face tracking for monitor arm control.

Hardware:
  Yaw servo  (horizontal pan)  → GPIO 18  (PWM, 400 Hz)
  Roll servo (vertical tilt)   → GPIO 13  (PWM, 400 Hz)
  Linear actuator (up/down)    → GPIO 20 (DIR) + GPIO 21 (EN)

Detection:
  YOLOv8n-face NCNN (WIDERFace) via raw ncnn bindings — no Ultralytics overhead
  Source: MJPEG stream via background thread (always latest frame)

Control:
  PID controller on face error → smooth servo movement (no jitter)
  EMA smoothing on raw face coordinates
  Deadzone prevents micro-corrections when face is near centre
  Linear actuator: timed pulses when vertical error is large

Usage:
  python monitor_arm_track.py
  python monitor_arm_track.py --source https://192.168.1.111:8002/mjpeg --insecure
  python monitor_arm_track.py --no-gpio   # test on desktop (no hardware)

Tuning:
  --kp / --ki / --kd     PID gains (start with defaults, increase kp for faster response)
  --deadzone-x/y         Fraction of frame (0.0-0.5) before servo responds
  --ema                  Smoothing factor (0=no smooth, 0.9=very smooth/slow)
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np
import requests
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# ── Default Parameters ────────────────────────────────────────────────────────

DEFAULT_MODEL   = "models/yolov8n-face_ncnn_model"
# Loopback by default: the MJPEG server runs on this same Pi, and loopback
# survives DHCP/network changes. Pass --source to override for remote streams.
DEFAULT_SOURCE  = "https://127.0.0.1:8002/mjpeg"
DEFAULT_CONF    = 0.45
RES_W, RES_H    = 640, 480

# GPIO pins
YAW_PIN     = 18   # servo: horizontal pan
ROLL_PIN    = 13   # servo: vertical tilt
ACT_DIR_PIN = 20   # linear actuator: direction (HR8833)
ACT_EN_PIN  = 21   # linear actuator: enable

# Servo PWM
PWM_FREQ      = 400   # Hz
SERVO_MIN     = 0.0
SERVO_MAX     = 1.0
SERVO_NEUTRAL = 0.5
SERVO_SLEEP_PITCH = 0.23    # pitch tilts down when going to sleep
SERVO_VERTICAL_START = 0.65   # pitch starting position (from face_detection_servo)
SERVO_RAMP_STEP     = 0.003   # duty change per ramp step
SERVO_RAMP_INTERVAL = 0.03    # seconds between ramp steps
MAX_SLEW            = 0.015   # max duty change per frame

# ── PID Profiles ─────────────────────────────────────────────────────────────
# Select with --profile <name>
PID_PROFILES = {
    "responsive": {     # Fast response, good for low-latency streams
        "kp": 0.06, "ki": 0.0002, "kd": 0.18,
        "ema": 0.65, "slew": 0.014,
        "dz_x": 0.16, "dz_y": 0.12,
    },
    "smooth": {         # Balanced — smooth but tracks well
        "kp": 0.04, "ki": 0.00015, "kd": 0.22,
        "ema": 0.80, "slew": 0.010,
        "dz_x": 0.18, "dz_y": 0.13,
    },
    "cinematic": {      # Slow — good for video/presentation
        "kp": 0.025, "ki": 0.0001, "kd": 0.30,
        "ema": 0.90, "slew": 0.006,
        "dz_x": 0.22, "dz_y": 0.15,
    },
}
DEFAULT_PROFILE = "smooth"

# Fallback defaults (used if no profile selected)
DEFAULT_KP = 0.04
DEFAULT_KI = 0.00015
DEFAULT_KD = 0.22

# Deadzone: fraction of frame dimension either side of centre
DEFAULT_DZ_X = 0.14
DEFAULT_DZ_Y = 0.10

# EMA smoothing on face position (higher = smoother but slower to respond)
DEFAULT_EMA = 0.80

# Camera offset: iPad camera is on the left side, so shift the tracking
# centre leftward. Fraction of frame width (positive = shift left).
CAMERA_OFFSET_X = 0.08

# Frames to hold last position after losing the face before resetting
HOLD_FRAMES = 25

# Centering nudge: once face has been in the deadzone for this many
# consecutive frames, slowly creep the servo toward true center.
CENTER_SETTLE_FRAMES = 8    # frames in deadzone before centering starts
CENTER_SPEED = 0.0008       # duty nudge per frame — very slow creep

# Linear actuator
ACT_DEADZONE_Y = 0.22   # only trigger actuator if Y error > this fraction of frame
ACT_PULSE_MS   = 220    # ms per actuator pulse
ACT_COOLDOWN_S = 1.2    # s between pulses (prevents oscillation)

# ── NCNN Face Detection ───────────────────────────────────────────────────────

INPUT_SIZE = 640

# Keypoint colours (BGR): left eye, right eye, nose, left mouth, right mouth
KPT_COLORS = [
    (255, 255,   0),  # left eye   — cyan
    (255, 255,   0),  # right eye  — cyan
    (  0, 255, 255),  # nose       — yellow
    (  0, 165, 255),  # left mouth — orange
    (  0, 165, 255),  # right mouth— orange
]


def load_model(model_dir: str) -> ncnn.Net:
    p = Path(model_dir)
    param = p / "model.ncnn.param"
    binf  = p / "model.ncnn.bin"
    if not param.exists() or not binf.exists():
        raise FileNotFoundError(f"NCNN model files not found in {p}\n"
                                f"Expected model.ncnn.param and model.ncnn.bin")
    net = ncnn.Net()
    net.opt.num_threads = 4   # use all 4 cores on Pi 5
    net.load_param(str(param))
    net.load_model(str(binf))
    return net


def detect_faces(net: ncnn.Net, frame: np.ndarray, conf_thresh: float) -> list[dict]:
    h, w = frame.shape[:2]
    scale  = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    pad_w  = (INPUT_SIZE - new_w) // 2
    pad_h  = (INPUT_SIZE - new_h) // 2

    mat_in = ncnn.Mat.from_pixels_resize(
        frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, new_w, new_h
    )
    mat_padded = ncnn.copy_make_border(
        mat_in,
        pad_h, INPUT_SIZE - new_h - pad_h,
        pad_w, INPUT_SIZE - new_w - pad_w,
        ncnn.BorderType.BORDER_CONSTANT, 114.0,
    )
    mat_padded.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])

    ex = net.create_extractor()
    ex.input("in0", mat_padded)
    _, out = ex.extract("out0")
    output = np.array(out)   # shape: (20, 8400)

    scores = output[4, :]
    mask   = scores > conf_thresh
    if not mask.any():
        return []

    filtered = output[:, mask]
    scores   = filtered[4, :]
    cx, cy   = filtered[0], filtered[1]
    bw, bh   = filtered[2], filtered[3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs  = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf_thresh, 0.45)
    if len(idxs) == 0:
        return []

    results = []
    for idx in idxs:
        i = int(np.array(idx).flat[0])
        dx1 = float(max(0, min(w, (x1[i] - pad_w) / scale)))
        dy1 = float(max(0, min(h, (y1[i] - pad_h) / scale)))
        dx2 = float(max(0, min(w, (x2[i] - pad_w) / scale)))
        dy2 = float(max(0, min(h, (y2[i] - pad_h) / scale)))
        # Extract 5 keypoints (x, y, visibility), each in model space → frame space
        kpts = []
        for k in range(5):
            kx_raw = float(filtered[5 + k * 3,     i])
            ky_raw = float(filtered[5 + k * 3 + 1, i])
            kv     = float(filtered[5 + k * 3 + 2, i])
            kx = float(max(0, min(w, (kx_raw - pad_w) / scale)))
            ky = float(max(0, min(h, (ky_raw - pad_h) / scale)))
            kpts.append((kx, ky, kv))
        results.append({
            "bbox": (int(dx1), int(dy1), int(dx2), int(dy2)),
            "conf": float(scores[i]),
            "area": (dx2 - dx1) * (dy2 - dy1),
            "keypoints": kpts,
        })
    return results


def pick_primary(detections: list[dict]) -> dict | None:
    """Return the largest (closest) detected face."""
    if not detections:
        return None
    return max(detections, key=lambda d: d["area"])


# ── PID Controller ────────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral  = 0.0
        self._prev_err  = 0.0
        self._limit     = 1.0   # anti-windup clamp on integral

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        self._integral = max(-self._limit, min(self._limit,
                             self._integral + error * dt))
        derivative    = (error - self._prev_err) / dt
        self._prev_err = error
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-1.0, min(1.0, out))

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0


# ── MJPEG Background Reader ───────────────────────────────────────────────────

class MjpegReader:
    """Reads MJPEG stream in a background thread. Always provides latest frame."""

    def __init__(self, url: str, res: tuple[int, int], insecure: bool):
        self.url       = url
        self.res       = res
        self.insecure  = insecure
        self._frame    = None
        self._frame_id = 0
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_frame(self) -> tuple[np.ndarray | None, int]:
        """Returns (frame, frame_id). Compare frame_id to detect new frames."""
        with self._lock:
            return self._frame, self._frame_id

    def _run(self):
        resW, resH = self.res
        while not self._stop.is_set():
            try:
                resp = requests.get(
                    self.url, stream=True,
                    verify=not self.insecure, timeout=(3, 10)
                )
                resp.raise_for_status()
                buf = b""
                for chunk in resp.iter_content(chunk_size=32768):
                    if self._stop.is_set():
                        return
                    buf += chunk
                    # Jump to the LAST complete JPEG in the buffer — discard backlogged frames
                    end = buf.rfind(b"\xff\xd9")
                    if end == -1:
                        continue
                    start = buf.rfind(b"\xff\xd8", 0, end)
                    if start == -1:
                        continue
                    jpg = buf[start:end + 2]
                    buf = buf[end + 2:]  # trim everything up to and including this frame
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        ih, iw = img.shape[:2]
                        if (iw, ih) != (resW, resH):
                            img = cv2.resize(img, (resW, resH), cv2.INTER_LINEAR)
                        with self._lock:
                            self._frame = img
                            self._frame_id += 1
            except Exception as e:
                if not self._stop.is_set():
                    print(f"[MJPEG] {e} — retrying in 2s")
                    time.sleep(2)


# ── Hardware PWM (sysfs) ─────────────────────────────────────────────────────

# RPi 5 RP1 PWM channel mapping (from `pinctrl funcs`)
#   GPIO 18 → PWM0_CHAN2 → pwmchip0/pwm2  (pinctrl alt: a3)
#   GPIO 13 → PWM0_CHAN1 → pwmchip0/pwm1  (pinctrl alt: a0)
HW_PWM_CHANNELS = {18: 2, 13: 1}
HW_PWM_PIN_ALT  = {18: "a3", 13: "a0"}

class HardwarePWM:
    """Hardware PWM via /sys/class/pwm (RP1 on RPi 5).
    Requires running as root: sysfs blocks non-root writes regardless of file
    permissions. The duty_cycle fd is held open to avoid an open()/close() per
    frame."""

    def __init__(self, gpio_pin: int, frequency: int = 400):
        channel = HW_PWM_CHANNELS.get(gpio_pin)
        if channel is None:
            raise ValueError(f"GPIO {gpio_pin} has no hardware PWM mapping")
        self._base = f"/sys/class/pwm/pwmchip0/pwm{channel}"
        self._period_ns = int(1e9 / frequency)
        self._value = 0.0

        # Set pin to PWM alt function via pinctrl
        alt = HW_PWM_PIN_ALT.get(gpio_pin)
        if alt:
            subprocess.run(["pinctrl", "set", str(gpio_pin), alt],
                           check=True, capture_output=True)

        # Export channel if not already
        if not Path(self._base).exists():
            Path("/sys/class/pwm/pwmchip0/export").write_text(str(channel))
            time.sleep(0.1)

        self._attr_write("period", str(self._period_ns))
        self._attr_write("duty_cycle", "0")
        self._attr_write("enable", "1")

        # Keep duty_cycle fd open for fast writes (no open/close per frame)
        self._duty_fd = open(f"{self._base}/duty_cycle", "w")

    def _attr_write(self, attr: str, val: str):
        with open(f"{self._base}/{attr}", "w") as f:
            f.write(val)
            f.flush()

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float):
        v = max(0.0, min(1.0, v))
        self._value = v
        self._duty_fd.seek(0)
        self._duty_fd.write(str(int(v * self._period_ns)))
        self._duty_fd.flush()

    def close(self):
        self._duty_fd.close()
        self._attr_write("enable", "0")


# ── Smooth Servo Interpolator ────────────────────────────────────────────────

class SmoothServo:
    """Inter-frame interpolator for servo PWM. Detection runs at ~4 FPS, so
    writing the PWM directly produces visibly stepped motion. A 200 Hz
    background thread chases the latest target in small steps. Same .value
    interface as HardwarePWM; the underlying PWM is exposed as .pwm for the
    shutdown ramp to bypass the interpolator."""

    def __init__(self, pwm, max_step: float = 0.0008, interval: float = 0.005):
        self.pwm       = pwm
        self._target   = pwm.value
        self._current  = pwm.value
        self._max_step = max_step
        self._interval = interval
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def value(self) -> float:
        return self._current

    @value.setter
    def value(self, v: float):
        with self._lock:
            self._target = max(0.0, min(1.0, v))

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            diff = target - self._current
            if abs(diff) > 1e-6:
                if abs(diff) <= self._max_step:
                    self._current = target
                else:
                    self._current += self._max_step if diff > 0 else -self._max_step
                self.pwm.value = self._current
            time.sleep(self._interval)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


# ── Linear Actuator ───────────────────────────────────────────────────────────

class LinearActuator:
    """HR8833 H-bridge control. Fires timed pulses with cooldown to prevent oscillation."""

    UP   =  1
    DOWN = -1

    def __init__(self, dir_pin: int, en_pin: int, factory):
        from gpiozero import DigitalOutputDevice
        self._dir  = DigitalOutputDevice(dir_pin, pin_factory=factory)
        self._en   = DigitalOutputDevice(en_pin,  pin_factory=factory)
        self._en.off()
        self._last = 0.0
        self._lock = threading.Lock()

    def trigger(self, direction: int, pulse_ms: float, cooldown_s: float):
        """Non-blocking: fires pulse in background thread if cooldown elapsed."""
        threading.Thread(
            target=self._pulse,
            args=(direction, pulse_ms, cooldown_s),
            daemon=True
        ).start()

    def _pulse(self, direction: int, pulse_ms: float, cooldown_s: float):
        now = time.monotonic()
        with self._lock:
            if now - self._last < cooldown_s:
                return
            self._last = now
        self._dir.value = (direction == self.UP)
        self._en.on()
        time.sleep(pulse_ms / 1000.0)
        self._en.off()

    def close(self):
        self._en.off()
        self._dir.close()
        self._en.close()


# ── Servo Ramp ────────────────────────────────────────────────────────────────

def move_servo_smoothly(pwm, start_duty: float, target_duty: float,
                        step: float = SERVO_RAMP_STEP,
                        interval: float = SERVO_RAMP_INTERVAL,
                        clamp_min: float = 0.1, clamp_max: float = 0.9) -> float:
    """Ramp a servo from start_duty to target_duty in small increments."""
    current = max(clamp_min, min(clamp_max, start_duty))
    target  = max(clamp_min, min(clamp_max, target_duty))
    pwm.value = current

    if abs(target - current) < 1e-9:
        return target

    direction = 1.0 if target > current else -1.0
    while (direction > 0 and current < target) or (direction < 0 and current > target):
        current += direction * step
        if (direction > 0 and current > target) or (direction < 0 and current < target):
            current = target
        pwm.value = current
        time.sleep(interval)

    return target


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monitor arm face tracking (YOLO NCNN + PID)")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="NCNN model directory")
    parser.add_argument("--source",      default=DEFAULT_SOURCE,
                        help="MJPEG stream URL")
    parser.add_argument("--conf",        type=float, default=DEFAULT_CONF,
                        help="Detection confidence threshold")
    parser.add_argument("--insecure",    action="store_true",
                        help="Skip TLS cert verification")
    parser.add_argument("--no-display",  action="store_true",
                        help="Disable preview window (headless)")
    parser.add_argument("--no-gpio",     action="store_true",
                        help="Disable GPIO (run on desktop for testing)")
    # PID profile
    parser.add_argument("--profile",     type=str, default=DEFAULT_PROFILE,
                        choices=list(PID_PROFILES.keys()),
                        help=f"PID tuning profile (default: {DEFAULT_PROFILE})")
    # PID tuning (overrides profile if set)
    parser.add_argument("--kp",          type=float, default=None)
    parser.add_argument("--ki",          type=float, default=None)
    parser.add_argument("--kd",          type=float, default=None)
    # Deadzone
    parser.add_argument("--deadzone-x",  type=float, default=None,
                        help="Horizontal deadzone fraction (0.0-0.5)")
    parser.add_argument("--deadzone-y",  type=float, default=None,
                        help="Vertical deadzone fraction (0.0-0.5)")
    # Smoothing
    parser.add_argument("--ema",         type=float, default=None,
                        help="EMA smoothing alpha (0=none, 0.9=heavy)")
    # Step-response data capture
    parser.add_argument("--log",         type=str, default=None,
                        help="Path to a CSV file to log per-frame state (for "
                             "step-response analysis). If unset, no logging is performed.")
    args = parser.parse_args()

    # Apply profile, then allow CLI overrides
    prof = PID_PROFILES[args.profile]
    if args.kp is None:         args.kp = prof["kp"]
    if args.ki is None:         args.ki = prof["ki"]
    if args.kd is None:         args.kd = prof["kd"]
    if args.ema is None:        args.ema = prof["ema"]
    if args.deadzone_x is None: args.deadzone_x = prof["dz_x"]
    if args.deadzone_y is None: args.deadzone_y = prof["dz_y"]
    global MAX_SLEW
    MAX_SLEW = prof["slew"]
    print(f"Profile: {args.profile} — kp={args.kp} ki={args.ki} kd={args.kd} "
          f"ema={args.ema} slew={MAX_SLEW} dz={args.deadzone_x}/{args.deadzone_y}")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / args.model
    print(f"Loading NCNN model from {model_path} ...")
    net = load_model(str(model_path))
    print("Model loaded.")

    # ── GPIO setup ────────────────────────────────────────────────────────────
    pwm_yaw = pwm_roll = actuator = factory = None
    if not args.no_gpio:
        from gpiozero.pins.lgpio import LGPIOFactory
        factory = LGPIOFactory()

        # Use hardware PWM if available (zero jitter), fall back to software
        try:
            pwm_yaw  = HardwarePWM(YAW_PIN,  PWM_FREQ)
            pwm_roll = HardwarePWM(ROLL_PIN, PWM_FREQ)
            print("Using HARDWARE PWM (sysfs)")
        except (OSError, PermissionError, ValueError) as e:
            print(f"Hardware PWM unavailable ({e}), falling back to software PWM")
            from gpiozero import PWMOutputDevice
            pwm_yaw  = PWMOutputDevice(YAW_PIN,  frequency=PWM_FREQ, pin_factory=factory)
            pwm_roll = PWMOutputDevice(ROLL_PIN, frequency=PWM_FREQ, pin_factory=factory)

        actuator = LinearActuator(ACT_DIR_PIN, ACT_EN_PIN, factory)
        pwm_yaw.value  = SERVO_NEUTRAL
        pwm_roll.value = SERVO_NEUTRAL
        print(f"GPIO ready — Yaw:GPIO{YAW_PIN}  Roll:GPIO{ROLL_PIN}  "
              f"Actuator:GPIO{ACT_DIR_PIN}+{ACT_EN_PIN}")
        # Gentle startup sweep to starting positions
        print("Sweeping servos to start position...")
        move_servo_smoothly(pwm_roll, SERVO_NEUTRAL, SERVO_VERTICAL_START)
        print("Ready.")

        # Wrap PWMs with the inter-frame interpolator. From here on, the main
        # loop sets a target with `pwm_*.value = ...` and the background
        # thread chases it at 200 Hz, eliminating the 250 ms "step + pause"
        # pattern caused by the slow ~4 FPS detection cadence.
        pwm_yaw  = SmoothServo(pwm_yaw)
        pwm_roll = SmoothServo(pwm_roll)
        print("Smooth servo interpolator running at 200 Hz.")

    # ── MJPEG stream ──────────────────────────────────────────────────────────
    reader = MjpegReader(args.source, (RES_W, RES_H), args.insecure)
    reader.start()
    print(f"Connecting to {args.source} ...")
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if reader.get_frame()[0] is not None:
            break
        time.sleep(0.1)
    else:
        print("ERROR: No frames received in 12s. Check MJPEG server is running.")
        reader.stop()
        sys.exit(1)
    print("Stream connected.")

    # ── PID controllers ───────────────────────────────────────────────────────
    pid_yaw  = PID(args.kp, args.ki, args.kd)
    pid_roll = PID(args.kp, args.ki, args.kd)

    # ── State ─────────────────────────────────────────────────────────────────
    duty_yaw   = SERVO_NEUTRAL
    duty_roll  = SERVO_NEUTRAL
    ema_x = ema_y = None
    hold_count    = 0
    dz_settle_x   = 0   # consecutive frames x-error has been in deadzone
    dz_settle_y   = 0
    t_prev        = time.monotonic()
    last_frame_id = -1

    print("Tracking started. Press Q in preview window or Ctrl+C to quit.")
    print(f"PID kp={args.kp} ki={args.ki} kd={args.kd}  "
          f"deadzone x={args.deadzone_x} y={args.deadzone_y}  ema={args.ema}")

    # ── Optional CSV logger for step-response capture ─────────────────────────
    log_writer = None
    log_file = None
    log_t0 = None
    if args.log:
        import csv as _csv
        log_file = open(args.log, "w", newline="")
        log_writer = _csv.writer(log_file)
        log_writer.writerow([
            "t_s", "raw_x", "raw_y", "ema_x", "ema_y",
            "duty_yaw", "duty_roll", "deadzone_x_px", "deadzone_y_px",
            "frame_w", "frame_h", "profile",
        ])
        log_t0 = time.monotonic()
        print(f"Logging per-frame state to {args.log}")

    try:
        while True:
            frame, frame_id = reader.get_frame()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = frame_id
            frame = cv2.flip(frame, 1)  # mirror horizontally

            frame  = frame.copy()
            fH, fW = frame.shape[:2]
            cx_mid = fW / 2.0 + (CAMERA_OFFSET_X * fW)
            cy_mid = fH / 2.0

            t_now  = time.monotonic()
            dt     = max(t_now - t_prev, 1e-4)
            t_prev = t_now

            # ── Detect faces ──────────────────────────────────────────────
            detections = detect_faces(net, frame, args.conf)
            primary    = pick_primary(detections)

            if primary:
                hold_count = 0
                x1, y1, x2, y2 = primary["bbox"]
                raw_x = (x1 + x2) / 2.0
                raw_y = (y1 + y2) / 2.0

                # EMA smoothing
                if ema_x is None:
                    ema_x, ema_y = raw_x, raw_y
                else:
                    a = args.ema
                    ema_x = (1 - a) * raw_x + a * ema_x
                    ema_y = (1 - a) * raw_y + a * ema_y

                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_text = f"{primary['conf']:.0%}"
                cv2.putText(frame, conf_text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame, (int(ema_x), int(ema_y)), 5, (0, 255, 255), -1)
                # Draw 5 facial keypoints
                for (kx, ky, kv), color in zip(primary["keypoints"], KPT_COLORS):
                    if kv > 0.3:
                        cv2.circle(frame, (int(kx), int(ky)), 4, color, -1)
            else:
                hold_count += 1
                if hold_count > HOLD_FRAMES:
                    # Face lost — reset smoothed position on next detection
                    ema_x = ema_y = None
                    pid_yaw.reset()
                    pid_roll.reset()


            # ── Servo + actuator control (PID + latency compensation) ────
            if ema_x is not None and not args.no_gpio:
                # Normalised error: -0.5 to +0.5
                err_x = (ema_x - cx_mid) / fW
                err_y = (ema_y - cy_mid) / fH

                # Apply deadzone — don't reset PID, just feed zero error
                # so the derivative term gently brakes instead of snapping back.
                # Once settled in deadzone, slowly nudge toward true center.
                raw_err_x = err_x   # keep original for centering
                raw_err_y = err_y
                if abs(err_x) < args.deadzone_x:
                    dz_settle_x += 1
                    err_x = 0.0
                else:
                    dz_settle_x = 0
                if abs(err_y) < args.deadzone_y:
                    dz_settle_y += 1
                    err_y = 0.0
                else:
                    dz_settle_y = 0

                # Latency compensation: if the servo is already moving
                # toward the face (error and last delta have same sign),
                # reduce the output — the stale frame hasn't caught up yet.
                raw_yaw  = pid_yaw.update(err_x, dt)
                raw_roll = pid_roll.update(err_y, dt)

                # If error is shrinking (derivative is opposite to error),
                # we're converging — apply a brake to prevent overshoot.
                # The derivative term in PID already does some of this,
                # but with latency we need extra dampening.
                CONVERGE_BRAKE = 0.4   # reduce output by this factor when converging
                if err_x != 0 and (err_x * pid_yaw._prev_err) > 0:
                    # Same sign → still chasing, check if error is shrinking
                    if abs(err_x) < abs(pid_yaw._prev_err):
                        raw_yaw *= CONVERGE_BRAKE
                if err_y != 0 and (err_y * pid_roll._prev_err) > 0:
                    if abs(err_y) < abs(pid_roll._prev_err):
                        raw_roll *= CONVERGE_BRAKE

                delta_yaw  = max(-MAX_SLEW, min(MAX_SLEW, raw_yaw))
                delta_roll = max(-MAX_SLEW, min(MAX_SLEW, raw_roll))

                duty_yaw  += delta_yaw
                duty_roll -= delta_roll   # inverted pitch

                # Centering nudge: face is in deadzone and settled — slowly
                # creep toward true center so the arm faces the user properly
                if dz_settle_x >= CENTER_SETTLE_FRAMES and abs(raw_err_x) > 0.01:
                    nudge_x = CENTER_SPEED if raw_err_x > 0 else -CENTER_SPEED
                    duty_yaw += nudge_x
                if dz_settle_y >= CENTER_SETTLE_FRAMES and abs(raw_err_y) > 0.01:
                    nudge_y = CENTER_SPEED if raw_err_y > 0 else -CENTER_SPEED
                    duty_roll -= nudge_y  # inverted pitch

                duty_yaw   = max(SERVO_MIN, min(SERVO_MAX, duty_yaw))
                duty_roll  = max(SERVO_MIN, min(SERVO_MAX, duty_roll))

                # Push the new target to the SmoothServo interpolator. The
                # background thread chases it at 200 Hz, so we don't need to
                # filter or threshold here — every update should flow through.
                pwm_yaw.value  = duty_yaw
                pwm_roll.value = duty_roll

                # ── CSV logging (step-response capture) ───────────────────────
                if log_writer is not None:
                    log_writer.writerow([
                        f"{time.monotonic() - log_t0:.4f}",
                        f"{raw_x:.2f}", f"{raw_y:.2f}",
                        f"{ema_x:.2f}", f"{ema_y:.2f}",
                        f"{duty_yaw:.5f}", f"{duty_roll:.5f}",
                        f"{args.deadzone_x * fW:.1f}",
                        f"{args.deadzone_y * fH:.1f}",
                        fW, fH, args.profile,
                    ])

                # Linear actuator: fires only when vertical error is large
                err_y = (ema_y - cy_mid) / fH
                if abs(err_y) > ACT_DEADZONE_Y:
                    direction = LinearActuator.DOWN if err_y > 0 else LinearActuator.UP
                    actuator.trigger(direction, ACT_PULSE_MS, ACT_COOLDOWN_S)

            # ── Preview window ────────────────────────────────────────────
            if not args.no_display:
                fps = 1.0 / dt
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                status = "TRACKING" if ema_x is not None else f"SEARCHING ({hold_count})"
                color  = (0, 255, 0) if ema_x is not None else (0, 165, 255)
                cv2.putText(frame, status, (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if not args.no_gpio and ema_x is not None:
                    cv2.putText(frame,
                                f"Yaw:{duty_yaw*100:.0f}%  Roll:{duty_roll*100:.0f}%",
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Centre crosshair
                cx_i, cy_i = int(cx_mid), int(cy_mid)
                cv2.line(frame, (cx_i - 15, cy_i), (cx_i + 15, cy_i), (200, 200, 200), 1)
                cv2.line(frame, (cx_i, cy_i - 15), (cx_i, cy_i + 15), (200, 200, 200), 1)

                # Deadzone box
                dz_px = int(args.deadzone_x * fW)
                dz_py = int(args.deadzone_y * fH)
                cv2.rectangle(frame,
                              (cx_i - dz_px, cy_i - dz_py),
                              (cx_i + dz_px, cy_i + dz_py),
                              (80, 80, 80), 1)

                cv2.imshow("Monitor Arm Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Shutting down...")
        if log_file is not None:
            log_file.close()
            print(f"Log written to {args.log}")
        reader.stop()
        if not args.no_gpio:
            # Stop the interpolator threads, then ramp the underlying PWMs
            # directly to the rest position. move_servo_smoothly() needs the
            # raw HardwarePWM, not the SmoothServo wrapper.
            print("Returning servos to neutral...")
            raw_yaw  = pwm_yaw.pwm
            raw_roll = pwm_roll.pwm
            pwm_yaw.stop()
            pwm_roll.stop()
            move_servo_smoothly(raw_yaw,  raw_yaw.value,  SERVO_NEUTRAL)
            move_servo_smoothly(raw_roll, raw_roll.value, SERVO_SLEEP_PITCH)
            time.sleep(0.2)
            raw_yaw.close()
            raw_roll.close()
            actuator.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
