#!/usr/bin/env python3
"""
Prepare face detection dataset for YOLO training.

This script:
1. Takes unlabeled face images from input directory
2. Auto-labels them using a pre-trained face detector
3. Generates YOLO format annotations
4. Creates train/val splits
5. Organizes everything into YOLO training structure

Usage:
    python prepare_face_dataset.py --input-dir "Face recognition/training" --output-dir "face_detection_dataset"
"""

import os
import sys
import argparse
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_face_cascade():
    """Load Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
    return face_cascade


def detect_faces_in_image(image_path, face_cascade, scale_factor=1.1, min_neighbors=5, min_size=(20, 20)):
    """
    Detect faces in an image and return normalized YOLO format annotations.
    
    Args:
        image_path: Path to image file
        face_cascade: OpenCV cascade classifier
        scale_factor: Detection scale factor
        min_neighbors: Minimum neighbors for detection
        min_size: Minimum face size
    
    Returns:
        List of YOLO format annotations: [(cx, cy, w, h), ...]
        Returns empty list if image can't be read
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        if len(faces) == 0:
            return []
        
        height, width = img.shape[:2]
        annotations = []
        
        for (x, y, w, h) in faces:
            # Convert to YOLO format (normalized center coordinates and width/height)
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height
            
            # Clamp values to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            annotations.append((center_x, center_y, norm_w, norm_h))
        
        return annotations
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


def save_yolo_annotation(output_file, annotations):
    """Save annotations in YOLO format."""
    with open(output_file, 'w') as f:
        for ann in annotations:
            f.write(f"0 {ann[0]:.6f} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f}\n")


def process_dataset(input_dir, output_dir, train_split=0.8):
    """
    Process dataset from input directory and create YOLO format dataset.
    
    Args:
        input_dir: Input directory containing subdirectories with images
        output_dir: Output directory for YOLO format dataset
        train_split: Proportion of data to use for training (0.8 = 80/20 split)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    train_images_dir = output_path / 'images' / 'train'
    val_images_dir = output_path / 'images' / 'val'
    train_labels_dir = output_path / 'labels' / 'train'
    val_labels_dir = output_path / 'labels' / 'val'
    
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load face detector
    print("Loading face detector...")
    face_cascade = get_face_cascade()
    
    # Collect all images
    print("Collecting images from input directory...")
    all_images = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print(f"Error: No images found in {input_dir}")
        return False
    
    print(f"Found {len(all_images)} images")
    
    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_split)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Train: {len(train_images)} images, Validation: {len(val_images)} images")
    
    # Process training images
    print("\nProcessing training images...")
    successful = 0
    for image_path in tqdm(train_images):
        annotations = detect_faces_in_image(image_path, face_cascade)
        
        if len(annotations) > 0:
            # Copy image
            filename = Path(image_path).name
            dst_image = train_images_dir / filename
            shutil.copy2(image_path, dst_image)
            
            # Save annotation
            label_file = train_labels_dir / (Path(filename).stem + '.txt')
            save_yolo_annotation(label_file, annotations)
            successful += 1
    
    print(f"Successfully processed {successful}/{len(train_images)} training images")
    
    # Process validation images
    print("\nProcessing validation images...")
    successful = 0
    for image_path in tqdm(val_images):
        annotations = detect_faces_in_image(image_path, face_cascade)
        
        if len(annotations) > 0:
            # Copy image
            filename = Path(image_path).name
            dst_image = val_images_dir / filename
            shutil.copy2(image_path, dst_image)
            
            # Save annotation
            label_file = val_labels_dir / (Path(filename).stem + '.txt')
            save_yolo_annotation(label_file, annotations)
            successful += 1
    
    print(f"Successfully processed {successful}/{len(val_images)} validation images")
    
    # Create data.yaml
    print("\nCreating data.yaml configuration...")
    data_yaml_path = output_path / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {output_path.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['face']\n")
    
    print(f"\nDataset created successfully!")
    print(f"Output directory: {output_path}")
    print(f"Data config: {data_yaml_path}")
    print(f"\nNext step: Train model with:")
    print(f"  yolo detect train data={data_yaml_path} model=yolov8n.pt epochs=100")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare face detection dataset for YOLO training'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory containing images (can have subdirectories)'
    )
    parser.add_argument(
        '--output-dir',
        default='face_detection_dataset',
        help='Output directory for YOLO format dataset (default: face_detection_dataset)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Proportion for train/val split (default: 0.8 for 80/20)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    if not 0 < args.train_split < 1:
        print("Error: train-split must be between 0 and 1")
        return 1
    
    try:
        success = process_dataset(args.input_dir, args.output_dir, args.train_split)
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
