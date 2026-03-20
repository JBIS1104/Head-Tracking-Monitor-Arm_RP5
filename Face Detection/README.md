# Face Detection Setup (venv)

This folder contains:
- `face_detection_regent_style.py`
- `face_detection_stream.py`
- `face_detection_servo.py`
- `face_detection_yolo_face.py`

## 1) Create and activate virtual environment

```bash
cd "/home/billyp/Documents/Face Detection"
python3 -m venv .venv-gui
source .venv-gui/bin/activate
python -m pip install --upgrade pip
```

## 2) Install Python libraries

```bash
pip install -r requirements.txt
```

## 3) Run scripts

```bash
python face_detection_regent_style.py
python face_detection_stream.py --show
python face_detection_servo.py --insecure
python face_detection_yolo_face.py --url "https://192.168.1.111:8002/mjpeg" --insecure
```

## Notes for Raspberry Pi

- `gpiozero` + `lgpio` are required for the servo script.
- If OpenCV wheel install fails, install system deps first:

```bash
sudo apt update
sudo apt install -y python3-opencv libatlas-base-dev
```

Then retry in the venv:

```bash
pip install -r requirements.txt
```

## Comments at top of scripts

The first lines in each script now contain bash commands showing how to activate the venv and run that script.
