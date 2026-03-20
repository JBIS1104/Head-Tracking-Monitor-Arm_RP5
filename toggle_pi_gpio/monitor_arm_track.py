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
DEFAULT_SOURCE  = "https://192.168.1.111:8002/mjpeg"
DEFAULT_CONF    = 0.45
RES_W, RES_H    = 640, 480

# GPIO pins
YAW_PIN     = 18   # servo: horizontal pan
ROLL_PIN    = 13   # servo: vertical tilt
ACT_DIR_PIN = 20   # linear actuator: direction (HR8833)
ACT_EN_PIN  = 21   # linear actuator: enable

# Servo PWM
PWM_FREQ      = 400   # Hz — matches existing working setup
SERVO_MIN     = 0.0
SERVO_MAX     = 1.0
SERVO_NEUTRAL = 0.5

# PID defaults (tune via command line)
DEFAULT_KP = 0.35
DEFAULT_KI = 0.0002
DEFAULT_KD = 0.10

# Deadzone: fraction of frame dimension either side of centre
DEFAULT_DZ_X = 0.08
DEFAULT_DZ_Y = 0.08

# EMA smoothing on face position (higher = smoother but slower to respond)
DEFAULT_EMA = 0.4

# Frames to hold last position after losing the face before resetting
HOLD_FRAMES = 25

# Linear actuator
ACT_DEADZONE_Y = 0.22   # only trigger actuator if Y error > this fraction of frame
ACT_PULSE_MS   = 220    # ms per actuator pulse
ACT_COOLDOWN_S = 1.2    # s between pulses (prevents oscillation)

# ── NCNN Face Detection ───────────────────────────────────────────────────────

INPUT_SIZE = 640


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
        results.append({
            "bbox": (int(dx1), int(dy1), int(dx2), int(dy2)),
            "conf": float(scores[i]),
            "area": (dx2 - dx1) * (dy2 - dy1),
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
        self.url      = url
        self.res      = res
        self.insecure = insecure
        self._frame   = None
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def _run(self):
        resW, resH = self.res
        while not self._stop.is_set():
            try:
                resp = requests.get(
                    self.url, stream=True,
                    verify=not self.insecure, timeout=(3, 10)
                )
                resp.raise_for_status()
                buf, last_end = b"", 0
                for chunk in resp.iter_content(chunk_size=32768):
                    if self._stop.is_set():
                        return
                    buf += chunk
                    while True:
                        s = buf.find(b"\xff\xd8", last_end)
                        if s == -1:
                            if len(buf) > 512 * 1024:
                                buf = buf[-256 * 1024:]
                            last_end = 0
                            break
                        e = buf.find(b"\xff\xd9", s + 2)
                        if e == -1:
                            break
                        jpg = buf[s:e + 2]
                        last_end = e + 2
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            ih, iw = img.shape[:2]
                            if (iw, ih) != (resW, resH):
                                img = cv2.resize(img, (resW, resH), cv2.INTER_LINEAR)
                            with self._lock:
                                self._frame = img
            except Exception as e:
                if not self._stop.is_set():
                    print(f"[MJPEG] {e} — retrying in 2s")
                    time.sleep(2)


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
    # PID tuning
    parser.add_argument("--kp",          type=float, default=DEFAULT_KP)
    parser.add_argument("--ki",          type=float, default=DEFAULT_KI)
    parser.add_argument("--kd",          type=float, default=DEFAULT_KD)
    # Deadzone
    parser.add_argument("--deadzone-x",  type=float, default=DEFAULT_DZ_X,
                        help="Horizontal deadzone fraction (0.0-0.5)")
    parser.add_argument("--deadzone-y",  type=float, default=DEFAULT_DZ_Y,
                        help="Vertical deadzone fraction (0.0-0.5)")
    # Smoothing
    parser.add_argument("--ema",         type=float, default=DEFAULT_EMA,
                        help="EMA smoothing alpha (0=none, 0.9=heavy)")
    args = parser.parse_args()

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
        from gpiozero import PWMOutputDevice
        from gpiozero.pins.lgpio import LGPIOFactory
        factory  = LGPIOFactory()
        pwm_yaw  = PWMOutputDevice(YAW_PIN,  frequency=PWM_FREQ, pin_factory=factory)
        pwm_roll = PWMOutputDevice(ROLL_PIN, frequency=PWM_FREQ, pin_factory=factory)
        actuator = LinearActuator(ACT_DIR_PIN, ACT_EN_PIN, factory)
        pwm_yaw.value  = SERVO_NEUTRAL
        pwm_roll.value = SERVO_NEUTRAL
        print(f"GPIO ready — Yaw:GPIO{YAW_PIN}  Roll:GPIO{ROLL_PIN}  "
              f"Actuator:GPIO{ACT_DIR_PIN}+{ACT_EN_PIN}")

    # ── MJPEG stream ──────────────────────────────────────────────────────────
    reader = MjpegReader(args.source, (RES_W, RES_H), args.insecure)
    reader.start()
    print(f"Connecting to {args.source} ...")
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if reader.get_frame() is not None:
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
    t_prev        = time.monotonic()

    print("Tracking started. Press Q in preview window or Ctrl+C to quit.")
    print(f"PID kp={args.kp} ki={args.ki} kd={args.kd}  "
          f"deadzone x={args.deadzone_x} y={args.deadzone_y}  ema={args.ema}")

    try:
        while True:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame  = frame.copy()
            fH, fW = frame.shape[:2]
            cx_mid = fW / 2.0
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
                    ema_x = a * raw_x + (1 - a) * ema_x
                    ema_y = a * raw_y + (1 - a) * ema_y

                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_text = f"{primary['conf']:.0%}"
                cv2.putText(frame, conf_text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame, (int(ema_x), int(ema_y)), 5, (0, 255, 255), -1)
            else:
                hold_count += 1
                if hold_count > HOLD_FRAMES:
                    # Face lost — reset smoothed position on next detection
                    ema_x = ema_y = None
                    pid_yaw.reset()
                    pid_roll.reset()

            # ── Servo + actuator control ──────────────────────────────────
            if ema_x is not None and not args.no_gpio:
                # Normalised error: -0.5 to +0.5
                # Positive err_x = face is to the right  → pan right (increase duty)
                # Positive err_y = face is below centre  → tilt down (increase duty)
                err_x = (ema_x - cx_mid) / fW
                err_y = (ema_y - cy_mid) / fH

                # Apply deadzone (zero out small errors)
                if abs(err_x) < args.deadzone_x:
                    err_x = 0.0
                    pid_yaw.reset()
                if abs(err_y) < args.deadzone_y:
                    err_y = 0.0
                    pid_roll.reset()

                # PID update → duty delta
                duty_yaw  += pid_yaw.update(err_x, dt)
                duty_roll += pid_roll.update(err_y, dt)
                duty_yaw   = max(SERVO_MIN, min(SERVO_MAX, duty_yaw))
                duty_roll  = max(SERVO_MIN, min(SERVO_MAX, duty_roll))

                pwm_yaw.value  = duty_yaw
                pwm_roll.value = duty_roll

                # Linear actuator: fires only when vertical error is large
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
        reader.stop()
        if not args.no_gpio:
            pwm_yaw.value  = SERVO_NEUTRAL
            pwm_roll.value = SERVO_NEUTRAL
            time.sleep(0.3)
            pwm_yaw.close()
            pwm_roll.close()
            actuator.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
