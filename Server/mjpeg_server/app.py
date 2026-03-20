#cd "/home/billyp/Documents/Server" && python -c "from mjpeg_server.app import app; app.run(host='0.0.0.0', port=8002, threaded=True, ssl_context='adhoc')"
#ps -p $(pgrep -f "mjpeg_server/app.py") -o %cpu,%mem,rss,vsz,cmd
from __future__ import annotations

from pathlib import Path
import threading
import time
from flask import Flask, Response, request, send_from_directory

WEB_ROOT = Path(__file__).resolve().parents[1] / "Billy_P_Website" / "VideoStreamApp"

app = Flask(__name__, static_folder=str(WEB_ROOT), static_url_path="")

LATEST_FRAME = {
    "bytes": None,
    "timestamp": 0.0,
}
FRAME_LOCK = threading.Lock()
BOUNDARY = "frame"


@app.after_request
def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_frame() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)

    data = request.get_data()
    if not data:
        return Response("No frame data", status=400)

    with FRAME_LOCK:
        LATEST_FRAME["bytes"] = data
        LATEST_FRAME["timestamp"] = time.time()

    return Response("OK", status=200)


@app.route("/mjpeg")
def mjpeg_stream() -> Response:
    def generate():
        last_timestamp = 0.0
        while True:
            with FRAME_LOCK:
                frame = LATEST_FRAME["bytes"]
                timestamp = LATEST_FRAME["timestamp"]

            if frame and timestamp != last_timestamp:
                last_timestamp = timestamp
                header = (
                    f"--{BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode("ascii")
                yield header + frame + b"\r\n"
            else:
                time.sleep(0.01)

    return Response(
        generate(),
        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


@app.route("/status")
def status() -> Response:
    with FRAME_LOCK:
        age = time.time() - LATEST_FRAME["timestamp"] if LATEST_FRAME["bytes"] else None
    if age is None:
        return Response("no frames", status=200)
    return Response(f"last frame: {age:.2f}s ago", status=200)


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path: str):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True, ssl_context="adhoc")
