"""
Local ultralytics shim for Raspberry Pi (no PyTorch required).
Backed by ncnn inference — works with system opencv (any version) for display.
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import ncnn


class _FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32)
    def cpu(self): return self
    def numpy(self): return self._data
    def squeeze(self): return self._data.squeeze()
    def item(self): return float(self._data.flat[0])
    def __len__(self): return len(self._data)


class Box:
    def __init__(self, x1, y1, x2, y2, conf, cls_idx):
        self.xyxy = _FakeTensor([[x1, y1, x2, y2]])
        self.conf = _FakeTensor([conf])
        self.cls  = _FakeTensor([float(cls_idx)])


class Boxes(list):
    pass


class Results:
    def __init__(self, boxes):
        self.boxes = boxes


def _postprocess(raw, conf_thresh, orig_w, orig_h, input_size=640):
    # raw shape: [rows, anchors] or [anchors, rows]
    # rows = 4 (bbox) + 1 (conf) + N*3 (keypoints)
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
        raw = raw.T  # -> [anchors, rows]

    scale = min(input_size / orig_h, input_size / orig_w)
    pad_w = (input_size - orig_w * scale) / 2
    pad_h = (input_size - orig_h * scale) / 2

    candidates = []
    for row in raw:
        cx, cy, w, h = row[:4]
        conf = float(row[4])
        if conf < conf_thresh:
            continue
        x1 = max(0.0, (cx - w/2 - pad_w) / scale)
        y1 = max(0.0, (cy - h/2 - pad_h) / scale)
        x2 = min(float(orig_w), (cx + w/2 - pad_w) / scale)
        y2 = min(float(orig_h), (cy + h/2 - pad_h) / scale)
        if x2 > x1 and y2 > y1:
            candidates.append((conf, 0, x1, y1, x2, y2))

    boxes = Boxes()
    if not candidates:
        return boxes

    # Only apply NMS if we have many detections (improves speed for sparse detections)
    if len(candidates) <= 3:
        for conf, cls_idx, x1, y1, x2, y2 in candidates:
            boxes.append(Box(x1, y1, x2, y2, conf, cls_idx))
        return boxes

    coords = [[c[2], c[3], c[4], c[5]] for c in candidates]
    scores = [c[0] for c in candidates]
    indices = cv2.dnn.NMSBoxes(coords, scores, conf_thresh, 0.45)
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif indices is None:
        indices = []
    for i in indices:
        conf, cls_idx, x1, y1, x2, y2 = candidates[i]
        boxes.append(Box(x1, y1, x2, y2, conf, cls_idx))
    return boxes


class YOLO:
    """Drop-in ultralytics.YOLO backed by ncnn — no PyTorch needed."""

    def __init__(self, model_path: str, task: str = "detect"):
        p = Path(model_path)
        ncnn_dir = p if p.is_dir() else p.parent / (p.stem + "_ncnn_model")
        if not ncnn_dir.exists():
            raise FileNotFoundError(f"No NCNN folder at {ncnn_dir}")

        param   = ncnn_dir / "model.ncnn.param"
        weights = ncnn_dir / "model.ncnn.bin"
        if not param.exists() or not weights.exists():
            raise FileNotFoundError(f"Missing .param or .bin in {ncnn_dir}")

        print(f"[ultralytics shim] Loading NCNN model: {ncnn_dir}")
        self._net = ncnn.Net()
        self._net.opt.use_vulkan_compute = True  # Enable GPU if available
        self._net.opt.num_threads = 4            # Use 4 threads (Pi5 has 4 cores)
        self._net.load_param(str(param))
        self._net.load_model(str(weights))
        self._input_size = 640
        self.names = {0: "face"}
        print("[ultralytics shim] NCNN model ready (GPU enabled if available, 4 threads).")
        
        # Reusable extractor for faster inference
        self._extractor_cache = None

    def _infer(self, frame, conf_thresh=0.5):
        orig_h, orig_w = frame.shape[:2]
        ex = self._net.create_extractor()
        mat = ncnn.Mat.from_pixels_resize(
            frame, ncnn.Mat.PixelType.PIXEL_BGR,
            orig_w, orig_h, self._input_size, self._input_size)
        mat.substract_mean_normalize([], [1/255.0]*3)
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        raw = np.array(out)
        return Results(_postprocess(raw, conf_thresh, orig_w, orig_h, self._input_size))

    def __call__(self, frame, verbose=False, conf=0.5, **kw):
        return [self._infer(frame, conf)]

    def track(self, frame, verbose=False, conf=0.5, **kw):
        return [self._infer(frame, conf)]

    def predict(self, frame, verbose=False, conf=0.5, **kw):
        return [self._infer(frame, conf)]
