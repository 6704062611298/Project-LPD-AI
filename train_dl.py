"""
train_dl.py - Deep Learning (YOLOv8) Model Training

Trains a YOLOv8 model for license plate detection.

Configuration:
- Model: YOLOv8n (nano — lightweight, suitable for deployment)
- Epochs: 50+
- Image size: 640×640
- Class: 0 = license_plate

Usage:
    python train_dl.py

Input:
    dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/

Output:
    models/yolo_best.pt — best model weights
"""

import os
import sys
import yaml
import shutil


# ===================== Configuration =====================
DATASET_DIR = "dataset"
DATA_YAML_PATH = "data.yaml"
MODEL_SAVE_DIR = "models"
YOLO_BEST_FILENAME = "yolo_best.pt"

# Training hyperparameters
EPOCHS = 10             # Number of training epochs (reduced for faster training)
IMG_SIZE = 416          # Input image size (smaller = faster)
BATCH_SIZE = 32         # Batch size (larger = faster, adjust if OOM)
MODEL_VARIANT = "yolov8n.pt"   # YOLOv8 nano (pretrained on COCO)
# =========================================================


def create_data_yaml():
    """
    Dynamically generates the data.yaml configuration file for YOLOv8.
    
    The YAML file specifies:
    - Absolute path to the dataset root
    - Relative paths to train and val image directories
    - Class names and IDs
    
    Returns:
        str: Path to the created data.yaml file.
    """
    dataset_abs_path = os.path.abspath(DATASET_DIR)
    
    data_config = {
        'path': dataset_abs_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'license_plate'
        }
    }
    
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False, default_flow_style=False)
    
    print(f"  Created '{DATA_YAML_PATH}' with dataset path: {dataset_abs_path}")
    return DATA_YAML_PATH


def check_dataset():
    """
    Validates that the dataset directories exist and contain files.
    
    Returns:
        bool: True if dataset is ready, False otherwise.
    """
    train_img_dir = os.path.join(DATASET_DIR, 'images', 'train')
    val_img_dir = os.path.join(DATASET_DIR, 'images', 'val')
    
    if not os.path.exists(train_img_dir):
        print(f"Error: Training images not found at '{train_img_dir}'")
        print("Please run prepare_dataset.py first!")
        return False
    
    train_images = [f for f in os.listdir(train_img_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not train_images:
        print(f"Error: No images found in '{train_img_dir}'")
        return False
    
    print(f"  Training images found: {len(train_images)}")
    
    if os.path.exists(val_img_dir):
        val_images = [f for f in os.listdir(val_img_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Validation images found: {len(val_images)}")
    
    return True


def train():
    """
    Main training function for YOLOv8 license plate detection.
    
    Steps:
    1. Check dataset availability
    2. Create data.yaml configuration
    3. Load pretrained YOLOv8n model
    4. Train with specified hyperparameters
    5. Copy best weights to models/ directory
    """
    print("=" * 60)
    print("  LICENSE PLATE DETECTION — YOLOv8 Training")
    print("=" * 60)
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nError: ultralytics package not installed!")
        print("Please run: pip install ultralytics")
        return
    
    # Step 1: Check dataset
    print("\n[Step 1] Checking dataset...")
    if not check_dataset():
        return
    
    # Step 2: Create data.yaml
    print("\n[Step 2] Creating data.yaml configuration...")
    yaml_path = create_data_yaml()
    
    # Step 3: Load pretrained model
    print(f"\n[Step 3] Loading pretrained model: {MODEL_VARIANT}")
    model = YOLO(MODEL_VARIANT)
    print("  Model loaded successfully!")
    
    # Step 4: Train the model
    print(f"\n[Step 4] Starting training...")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Augmentations: rotation ± 10°, brightness ± 20%")
    
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project='runs',
        name='detect/license_plate',
        
        # Built-in augmentation parameters
        degrees=10.0,         # Rotation augmentation (± degrees)
        hsv_v=0.2,            # Brightness (Value channel in HSV)

        flipud=0.0,           # Disable vertical flip (plates aren't upside-down)
        fliplr=0.5,           # Horizontal flip
        
        exist_ok=True,        # Overwrite existing experiment
        verbose=True
    )
    
    # Step 5: Copy best model weights to models/ directory
    print(f"\n[Step 5] Saving best model weights...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Use the trainer's save_dir to find the actual best.pt path
    best_pt_path = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
    dest_path = os.path.join(MODEL_SAVE_DIR, YOLO_BEST_FILENAME)
    
    if os.path.exists(best_pt_path):
        shutil.copy(best_pt_path, dest_path)
        print(f"  Best model saved to: {dest_path}")
    else:
        print(f"  Warning: best.pt not found at: {best_pt_path}")
        print("  Check runs/detect/license_plate/ for trained weights")
    
    print("\n" + "=" * 60)
    print("  YOLOv8 Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    train()
