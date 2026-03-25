"""
train_ml.py - Machine Learning Ensemble Model Training

Trains a license plate classification ensemble using traditional ML algorithms:
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

These are combined using a Voting Classifier (soft voting).

Features extracted: color histogram, edge features, texture (LBP), HOG.

Usage:
    python train_ml.py

Input:
    dataset/images/train/ — training images with corresponding labels
    dataset/images/val/   — validation images with corresponding labels

Output:
    models/ml_ensemble_model.pkl — trained ensemble model
"""

import os
import sys
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.feature_extraction import extract_all_features


# ===================== Configuration =====================
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "models/ml_ensemble_model.pkl"
PATCH_SIZE = (128, 128)   # Standard size for feature extraction
RANDOM_STATE = 42
# =========================================================


def load_dataset(split='train'):
    """
    Loads image patches and labels for ML training.
    
    For each image, extracts:
    - Positive samples: cropped license plate regions (label=1)
    - Negative samples: random non-plate regions (label=0)
    
    Args:
        split: 'train' or 'val'.
        
    Returns:
        tuple: (features_array, labels_array) as numpy arrays.
    """
    images_dir = os.path.join(DATASET_DIR, 'images', split)
    labels_dir = os.path.join(DATASET_DIR, 'labels', split)
    
    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}")
        return np.array([]), np.array([])
    
    features_list = []
    labels_list = []
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    
    print(f"  Loading {split} set: {len(image_files)} images...")
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, base_name + '.txt')
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Parse YOLO-format labels
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                try:
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = [float(x) for x in parts[1:5]]
                except ValueError:
                    continue
                
                if cls_id != 0:  # Only class 0 = license_plate
                    continue
                
                # Convert normalized coords to pixel coords
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)
                
                # Clamp to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Crop positive sample (license plate region)
                plate_crop = image[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue
                
                plate_resized = cv2.resize(plate_crop, PATCH_SIZE)
                
                try:
                    feat = extract_all_features(plate_resized)
                    features_list.append(feat)
                    labels_list.append(1)  # 1 = License Plate
                except Exception:
                    continue
                
                # Generate negative sample (random non-plate region)
                for _ in range(2):  # 2 negatives per positive for balance
                    neg_x1 = np.random.randint(0, max(w - PATCH_SIZE[0], 1))
                    neg_y1 = np.random.randint(0, max(h - PATCH_SIZE[1], 1))
                    neg_x2 = neg_x1 + PATCH_SIZE[0]
                    neg_y2 = neg_y1 + PATCH_SIZE[1]
                    
                    # Check overlap with plate region — skip if overlapping
                    overlap_x = max(0, min(x2, neg_x2) - max(x1, neg_x1))
                    overlap_y = max(0, min(y2, neg_y2) - max(y1, neg_y1))
                    overlap_area = overlap_x * overlap_y
                    neg_area = PATCH_SIZE[0] * PATCH_SIZE[1]
                    
                    if overlap_area / max(neg_area, 1) > 0.3:
                        continue  # Too much overlap, skip
                    
                    neg_crop = image[neg_y1:neg_y2, neg_x1:neg_x2]
                    if neg_crop.size == 0:
                        continue
                    
                    neg_resized = cv2.resize(neg_crop, PATCH_SIZE)
                    
                    try:
                        feat = extract_all_features(neg_resized)
                        features_list.append(feat)
                        labels_list.append(0)  # 0 = No License Plate
                    except Exception:
                        continue
    
    if not features_list:
        return np.array([]), np.array([])
    
    return np.array(features_list), np.array(labels_list)


def build_ensemble():
    """
    Builds the ensemble model combining Random Forest, SVM, and KNN.
    
    Uses a VotingClassifier with soft voting to combine predictions
    from all three models with equal weight.
    
    Returns:
        Pipeline: Scikit-learn pipeline with StandardScaler + VotingClassifier.
    """
    # Model 1: Random Forest — good for non-linear boundaries
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Model 2: SVM — effective in high-dimensional spaces
    svm_clf = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,       # Required for soft voting
        random_state=RANDOM_STATE
    )
    
    # Model 3: KNN — simple instance-based learner
    knn_clf = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    
    # Combine with Voting Classifier (soft voting for probability-based ensemble)
    ensemble = VotingClassifier(
        estimators=[
            ('random_forest', rf_clf),
            ('svm', svm_clf),
            ('knn', knn_clf)
        ],
        voting='soft'
    )
    
    # Wrap in a pipeline with StandardScaler for feature normalization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', ensemble)
    ])
    
    return pipeline


def train():
    """
    Main training function:
    1. Loads train/val datasets
    2. Builds ensemble model
    3. Trains on training data
    4. Evaluates on validation data
    5. Saves trained model
    """
    print("=" * 60)
    print("  LICENSE PLATE DETECTION — ML Ensemble Training")
    print("=" * 60)
    
    # Step 1: Load datasets
    print("\n[Step 1] Loading training data...")
    X_train, y_train = load_dataset('train')
    
    print("\n[Step 2] Loading validation data...")
    X_val, y_val = load_dataset('val')
    
    if X_train.size == 0:
        print("\nError: No training data found!")
        print("Please run prepare_dataset.py first to create the dataset.")
        return
    
    print(f"\n  Training samples:   {len(X_train)} (Positive: {sum(y_train==1)}, Negative: {sum(y_train==0)})")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    if X_val.size > 0:
        print(f"  Validation samples: {len(X_val)} (Positive: {sum(y_val==1)}, Negative: {sum(y_val==0)})")
    
    # Step 2: Build ensemble model
    print("\n[Step 3] Building ensemble model (RF + SVM + KNN)...")
    model = build_ensemble()
    
    # Step 3: Train the model
    print("\n[Step 4] Training ensemble model...")
    model.fit(X_train, y_train)
    print("  Training complete!")
    
    # Step 4: Evaluate on training data
    print("\n[Step 5] Evaluating model...")
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n  Training Accuracy: {train_acc:.4f}")
    
    # Evaluate on validation data if available
    if X_val.size > 0:
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        print("\n  Validation Classification Report:")
        target_names = ['No Plate', 'License Plate']
        report = classification_report(y_val, val_pred, target_names=target_names)
        print(report)
    
    # Step 5: Save the trained model
    print(f"\n[Step 6] Saving model to '{MODEL_SAVE_PATH}'...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"  Model saved successfully!")
    
    print("\n" + "=" * 60)
    print("  ML Ensemble Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    train()
