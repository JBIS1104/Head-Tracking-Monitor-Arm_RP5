# Training a Custom Face Detection Model with YOLO

This guide shows how to train a custom YOLOv8/v11 face detection model using the EdjeElectronics methodology, adapted for face detection with your existing training data.

## Overview

You'll create a face detection model trained on your personal dataset of celebrities in the `Face recognition/training/` directory. This model will detect faces more accurately than pre-trained models for your specific use case.

## Prerequisites

- Python 3.8+ with pip
- NVIDIA GPU recommended (training will be much slower on CPU)
- Training images with face annotations (YOLO format)
- ~2-4 hours for training on GPU (~8+ hours on CPU)

## Step 1: Prepare Your Training Data

Your training data is already organized in:
```
Face recognition/training/
├── ben_affleck/
├── elon_musk/
├── elton_john/
├── jerry_seinfeld/
├── joe_biden/
├── junbyung_park/
├── madonna/
└── mindy_kaling/
```

### Option A: Use Existing Images with Auto-Labeling

If you have unlabeled face images, use a pre-trained face detector to automatically generate labels:

```bash
python prepare_face_dataset.py --input-dir "Face recognition/training" --output-dir "face_detection_dataset"
```

This script will:
1. Detect faces in each image using a pre-trained detector
2. Generate YOLO format annotations (.txt files)
3. Create the required folder structure for training
4. Split data into 80% train / 20% validation

### Option B: Use Label Studio (Recommended for Quality)

For maximum accuracy, manually label faces using Label Studio:

1. Install Label Studio:
   ```bash
   pip install label-studio
   ```

2. Start Label Studio:
   ```bash
   label-studio
   ```

3. Create a project:
   - Label Type: Object Detection
   - Import images from `Face recognition/training/`
   - Draw bounding boxes around each face
   - Export as YOLO format

4. Unzip exported data and organize:
   ```
   face_detection_dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

## Step 2: Create Data Configuration

Create `data.yaml` in your dataset folder:

```yaml
path: /path/to/face_detection_dataset
train: images/train
val: images/val
nc: 1
names: ['face']
```

## Step 3: Install Ultralytics

```bash
pip install ultralytics opencv-python
```

## Step 4: Train the Model

### On GPU (Recommended - 2-4 hours)

```bash
yolo detect train data=/path/to/data.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0
```

### On CPU (Slower - 8+ hours)

```bash
yolo detect train data=/path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640 device=cpu
```

### Training Parameters Explained

- **model**: YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)
  - Use `yolov8n.pt` for speed on Raspberry Pi
  - Use `yolov8s.pt` or `yolov8m.pt` for better accuracy
  
- **epochs**: Number of training passes through the dataset
  - 50-100 for face detection typically works well
  
- **imgsz**: Input image resolution
  - 640x640 is standard
  - Use 416x416 or 320x320 for faster training on limited GPU
  
- **batch**: Batch size (default 16)
  - Increase if you have GPU memory
  - `batch=32` or `batch=64` for faster training

### Example: Full Training Command

```bash
yolo detect train \
    data=face_detection_dataset/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    patience=20 \
    save=True \
    name=face_detector_v1
```

## Step 5: Monitor Training

Training results are saved in `runs/detect/train/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.png` - Training graphs (loss, precision, recall, mAP)
- `results.csv` - Metrics in CSV format

View results:
```bash
python -c "
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open('runs/detect/train/results.png')
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
"
```

## Step 6: Test Your Model

Test on validation images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=face_detection_dataset/images/val
```

View results in `runs/detect/predict/`

Test on live camera:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0 conf=0.5
```

## Step 7: Deploy to Raspberry Pi

### Convert to ONNX

```bash
# On your training machine (with GPU/CUDA)
yolo detect export model=runs/detect/train/weights/best.pt format=onnx
```

This creates `runs/detect/train/weights/best.onnx`

### Transfer to Raspberry Pi

```bash
scp runs/detect/train/weights/best.onnx billyp@192.168.1.111:/home/billyp/Documents/Face\ Detection/models/my_face_detector.onnx
```

### Run on Pi

```bash
python yolodesk live --model /path/to/my_face_detector.onnx --source 0 --conf 0.5
```

## Troubleshooting

### Model Not Learning (Loss stays high)

1. Check data: Verify images have correct face annotations
2. Increase epochs: Try 150-200 epochs
3. Increase data: Collect more training images (200+ minimum, 500+ recommended)
4. Use larger model: Try `yolov8s.pt` or `yolov8m.pt`

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
yolo detect train data=data.yaml model=yolov8n.pt batch=8 imgsz=416

# Or reduce image size
yolo detect train data=data.yaml model=yolov8n.pt imgsz=416
```

### Model Too Slow on Pi

1. Use nano model: `yolov8n.pt`
2. Lower confidence threshold: `--conf 0.6`
3. Export to NCNN format for faster inference
4. Skip frames during inference

### Model Missing Faces

1. Review training images for annotation errors
2. Increase number of epochs
3. Use pre-trained model with fine-tuning instead of training from scratch
4. Collect more diverse face images (different angles, lighting, ethnicities)

## Advanced: Transfer Learning (Faster Training)

Instead of training from scratch, start with a pre-trained face detection model:

```bash
# Fine-tune a pre-trained face model
yolo detect train \
    data=face_detection_dataset/data.yaml \
    model=yolov8s.pt \
    epochs=50 \
    freeze=10  # Freeze first 10 layers
```

This is much faster (10-20 minutes instead of hours) and often works better with limited data.

## Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **EdjeElectronics Guide**: https://www.ejtech.io/learn/train-yolo-models
- **Label Studio**: https://labelstud.io/
- **YOLO Face Detection**: Search "YOLOv8 face detection" for pre-trained models

## Next Steps

1. Prepare your dataset using Option A or B
2. Run training using the commands above
3. Export model to ONNX/NCNN format
4. Deploy to Raspberry Pi using the existing yolodesk setup
5. Monitor performance and fine-tune as needed
