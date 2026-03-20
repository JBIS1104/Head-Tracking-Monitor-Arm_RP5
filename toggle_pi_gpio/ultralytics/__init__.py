"""
Local ultralytics shim for Raspberry Pi (no PyTorch required).
Provides the YOLO API used by toggle_pi_gpio.py, backed by cv2.dnn ONNX inference.

Drop this folder alongside toggle_pi_gpio.py so Python finds it before the
installed (torch-requiring) ultralytics package.
"""

from .yolo import YOLO

__all__ = ["YOLO"]
