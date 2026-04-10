#!/usr/bin/env python3
"""
startup.py — One-command launcher for Face Tracking System

Usage:
  python3 startup.py              # with GPIO (servos)
  python3 startup.py --no-gpio   # without hardware (testing)
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent
VENV_DETECT = ROOT / "Face Detection" / ".venv-gui" / "bin" / "python"
VENV_SERVER = ROOT / "Server" / ".venv" / "bin" / "python"
TRACKER_DIR = ROOT / "toggle_pi_gpio"
SERVER_DIR  = ROOT / "Server"
SERVER_PORT = 8002

def detect_lan_ip() -> str:
    """Return this host's primary LAN IP (the one a remote client would reach)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send a packet — just picks the route to an external IP.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

PI_IP       = detect_lan_ip()
# Readiness probe hits loopback — the server listens on 0.0.0.0, and the LAN IP
# can shift between runs (DHCP), which used to cause spurious "did not start" errors.
STATUS_URL  = f"https://127.0.0.1:{SERVER_PORT}/status"

PROCS: dict[str, subprocess.Popen] = {}

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"

def banner(text: str):
    w = 60
    print(f"\n{CYAN}{'─' * w}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{'─' * w}{RESET}")

def ok(text: str):    print(f"{GREEN}  ✔  {text}{RESET}")
def warn(text: str):  print(f"{YELLOW}  ⚠  {text}{RESET}")
def err(text: str):   print(f"{RED}  ✘  {text}{RESET}")
def info(text: str):  print(f"{DIM}     {text}{RESET}")

# ── System stats ──────────────────────────────────────────────────────────────

def _stat_vals():
    with open("/proc/stat") as f:
        p = f.readline().split()
    v = list(map(int, p[1:8]))
    return sum(v), v[3] + v[4]

def cpu_pct(interval=0.4):
    t1, i1 = _stat_vals()
    time.sleep(interval)
    t2, i2 = _stat_vals()
    dt = t2 - t1
    return round(100 * (1 - (i2 - i1) / dt), 1) if dt else 0.0

def mem_info():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k.strip()] = int(v.split()[0])
    used = (info["MemTotal"] - info["MemAvailable"]) // 1024
    total = info["MemTotal"] // 1024
    return used, total, round(100 * used / total, 1)

def cpu_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"],
                                      text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        pass
    try:
        return float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:
        return None

def colour_val(val, warn_t, crit_t):
    if val >= crit_t:  return RED
    if val >= warn_t:  return YELLOW
    return GREEN

def bar(pct, width=16):
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)

def show_stats():
    banner("Step 1 — System Stats + Pre-flight")

    # Quick inline snapshot
    print(f"  Reading CPU (0.4s sample)...")
    cpu   = cpu_pct()
    used, total, mem_p = mem_info()
    temp  = cpu_temp()

    c = colour_val(cpu, 60, 85)
    print(f"  CPU   {c}{bar(cpu)}{RESET}  {c}{cpu:5.1f}%{RESET}")

    c = colour_val(mem_p, 60, 80)
    print(f"  MEM   {c}{bar(mem_p)}{RESET}  {c}{mem_p:5.1f}%{RESET}  ({used} / {total} MB)")

    if temp is not None:
        c = colour_val(temp, 60, 75)
        print(f"  TEMP  {c}{bar((temp/85)*100)}{RESET}  {c}{temp:.1f} °C{RESET}")

    # Pre-flight checks
    print()
    checks = [
        ("Face Detection venv",  VENV_DETECT.exists()),
        ("Server venv",          VENV_SERVER.exists()),
        ("NCNN model (.bin)",    (TRACKER_DIR / "models" / "yolov8n-face_ncnn_model" / "model.ncnn.bin").exists()),
        ("NCNN model (.param)",  (TRACKER_DIR / "models" / "yolov8n-face_ncnn_model" / "model.ncnn.param").exists()),
    ]
    all_ok = True
    for label, result in checks:
        if result:
            ok(label)
        else:
            err(label)
            all_ok = False

    if not all_ok:
        print()
        warn("Some checks failed. See RECOVERY.md to fix before continuing.")
        sys.exit(1)

    # Launch live stats dashboard in a separate terminal window
    stats_script = ROOT / "rpi_stats.py"
    if stats_script.exists():
        stats_cmd = ["lxterminal", "--title=RPi5 Stats",
                     "-e", f"python3 {stats_script}"]
        p = subprocess.Popen(stats_cmd, preexec_fn=os.setsid)
        PROCS["stats"] = p
        ok("Live stats dashboard opened in separate window")

# ── Process management ────────────────────────────────────────────────────────

def _pipe_output(proc: subprocess.Popen, tag: str, colour: str):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  {colour}[{tag}]{RESET} {DIM}{line}{RESET}")

def start_proc(name: str, cmd: list, cwd: Path, tag: str, colour: str):
    p = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        preexec_fn=os.setsid  # new process group so we can kill the whole tree
    )
    t = threading.Thread(target=_pipe_output, args=(p, tag, colour), daemon=True)
    t.start()
    PROCS[name] = p
    return p

# ── Cleanup ───────────────────────────────────────────────────────────────────

def kill_leftovers():
    """Kill leftover tracker/server processes from a previous run."""
    for pattern in ["monitor_arm_track", "mjpeg_server.app"]:
        try:
            old = subprocess.check_output(["pgrep", "-f", pattern], text=True).split()
            for pid in old:
                pid = int(pid.strip())
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
                    warn(f"Killed leftover {pattern} (pid {pid})")
            time.sleep(1)
        except (subprocess.CalledProcessError, ProcessLookupError, ValueError):
            pass

# ── MJPEG server ──────────────────────────────────────────────────────────────

def start_server():
    banner("Step 2 — MJPEG Server")

    if "server" in PROCS and PROCS["server"].poll() is None:
        warn("Server already running — restarting.")
        PROCS["server"].terminate()
        time.sleep(1)

    cmd = [
        str(VENV_SERVER), "-u", "-c",
        "from mjpeg_server.app import app; "
        "app.run(host='0.0.0.0', port=8002, threaded=True, ssl_context='adhoc')"
    ]
    start_proc("server", cmd, SERVER_DIR, "SERVER", CYAN)
    print(f"  Starting server on port {SERVER_PORT}...")

    # Wait until reachable
    try:
        import requests, urllib3
        urllib3.disable_warnings()
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests, urllib3
        urllib3.disable_warnings()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            r = requests.get(STATUS_URL, verify=False, timeout=2)
            if r.status_code == 200:
                ok(f"Server ready — {STATUS_URL}")
                return
        except Exception:
            pass
        time.sleep(1)
        print("  .", end="", flush=True)

    print()
    err("Server did not start in 20s. Check output above.")
    sys.exit(1)

# ── iPad wait ─────────────────────────────────────────────────────────────────

def wait_for_ipad():
    import requests
    banner("Step 3 — Waiting for iPad")
    print(f"  On your iPad, open the camera app and stream to:")
    print(f"  {YELLOW}{BOLD}  https://{PI_IP}:{SERVER_PORT}/upload{RESET}")
    print()

    SPIN = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    t_start = time.monotonic()

    while True:
        elapsed = time.monotonic() - t_start
        try:
            r = requests.get(STATUS_URL, verify=False, timeout=2)
            text = r.text.strip()
            if "last frame" in text:
                try:
                    age = float(text.split(":")[1].strip().split("s")[0])
                except Exception:
                    age = 999
                if age < 3.0:
                    print(f"\r  {GREEN}✔  iPad connected! Last frame {age:.2f}s ago (waited {elapsed:.0f}s){RESET}")
                    return
                else:
                    msg = f"Frame seen but stale ({age:.1f}s ago)"
            else:
                msg = "No frames yet"
        except Exception as e:
            msg = f"Server unreachable ({e})"

        spin = SPIN[i % len(SPIN)]
        print(f"\r  {CYAN}{spin}{RESET}  {msg}  [{elapsed:.0f}s]    ", end="", flush=True)
        i += 1
        time.sleep(1)

# ── Tracker ───────────────────────────────────────────────────────────────────

def start_tracker(no_gpio: bool):
    banner("Step 4 — Face Detection + Servo Tracking")

    # Run with sudo for hardware PWM sysfs access (required on RPi 5).
    # --source is forced to loopback: the MJPEG server runs on this same Pi,
    # and loopback survives any LAN/DHCP change (the tracker's hardcoded
    # default points at an old IP).
    cmd = [
        "sudo", str(VENV_DETECT), "-u", "monitor_arm_track.py",
        "--insecure",
        "--source", f"https://127.0.0.1:{SERVER_PORT}/mjpeg",
    ]
    if no_gpio:
        cmd.append("--no-gpio")
        warn("Running without GPIO (--no-gpio). Servos will NOT move.")

    start_proc("tracker", cmd, TRACKER_DIR, "TRACKER", GREEN)
    print(f"  Tracker starting...")
    time.sleep(3)

    if PROCS["tracker"].poll() is not None:
        err("Tracker exited early — check output above.")
        sys.exit(1)

    ok("Tracker running.")
    print()
    print(f"  {BOLD}Everything is up.{RESET}")
    print(f"  {DIM}Stream:  https://{PI_IP}:{SERVER_PORT}/mjpeg{RESET}")
    print(f"  {DIM}Status:  https://{PI_IP}:{SERVER_PORT}/status{RESET}")
    print()
    ok("Press  Ctrl+C  to shut everything down.")

# ── Shutdown ──────────────────────────────────────────────────────────────────

def shutdown():
    print()
    banner("Shutting down")
    for name, proc in PROCS.items():
        if proc.poll() is None:
            print(f"  Stopping {name} (pid {proc.pid})...")
            try:
                # Kill the entire process group (proc + all children)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
                ok(f"{name} stopped.")
            except (ProcessLookupError, PermissionError):
                ok(f"{name} already gone.")
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                warn(f"{name} force-killed.")
        else:
            info(f"{name} already stopped.")
    PROCS.clear()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face tracking system launcher")
    parser.add_argument("--no-gpio", action="store_true",
                        help="Run without servo hardware (testing)")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}  Face Tracking System — RPi5{RESET}")
    print(f"  {DIM}{ROOT}{RESET}")

    try:
        kill_leftovers()
        show_stats()
        start_server()
        wait_for_ipad()
        start_tracker(args.no_gpio)

        # Keep running, let threads stream output
        notified = set()
        while True:
            for name, proc in PROCS.items():
                if proc.poll() is not None and name not in notified:
                    warn(f"{name} exited unexpectedly (code {proc.returncode})")
                    notified.add(name)
            time.sleep(5)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
