"""
prepare_dataset.py - Automated Dataset Preparation Pipeline

This script handles the complete dataset preparation workflow:
1. Reads raw images and labels from raw_images/ and raw_labels/
2. Validates and cleans data (removes blurry, small plates, bad labels)
3. Generates augmented/synthetic dataset (Dataset 2)
4. Splits into 80% train / 20% validation
5. Creates structured YOLO-format dataset directory

Usage:
    python prepare_dataset.py

Input structure:
    raw_images/   <- Put your raw license plate images here
    raw_labels/   <- Put corresponding YOLO-format .txt labels here

Output structure:
    dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/
"""

import os
import sys
import shutil
import random
import cv2

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import is_blurry, clean_label_file, validate_image
from utils.augmentation import generate_augmented_dataset


# ===================== Configuration =====================
RAW_IMAGE_DIR = "raw_images"       # Source: real-world images (Dataset 1)
RAW_LABEL_DIR = "raw_labels"       # Source: corresponding YOLO labels
DATASET_DIR = "dataset"            # Output: structured dataset
AUGMENTED_TEMP_DIR = "temp_augmented"  # Temp dir for augmented data

TRAIN_SPLIT = 0.8                  # 80% train, 20% validation
NUM_AUGMENTATIONS = 3              # Number of augmented copies per image
RANDOM_SEED = 42                   # For reproducible splits
# =========================================================


def create_directory_structure():
    """
    Creates the YOLO-format dataset directory structure.
    
    Structure:
        dataset/
            images/train/
            images/val/
            labels/train/
            labels/val/
    """
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            path = os.path.join(DATASET_DIR, subdir, split)
            os.makedirs(path, exist_ok=True)
            print(f"  Created: {path}")


def collect_valid_pairs():
    """
    Scans raw_images/ and raw_labels/ directories for valid image-label pairs.
    
    Applies cleaning:
    - Skips images that can't be loaded
    - Skips blurry images
    - Filters invalid labels (wrong class, small plates, low contrast)
    
    Returns:
        list: List of dicts with keys 'image_path', 'label_lines', 'filename'.
    """
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"Error: '{RAW_IMAGE_DIR}' directory not found!")
        print(f"Please create '{RAW_IMAGE_DIR}/' and add your raw license plate images.")
        return []
    
    if not os.path.exists(RAW_LABEL_DIR):
        print(f"Warning: '{RAW_LABEL_DIR}' directory not found.")
        print(f"Creating '{RAW_LABEL_DIR}/' — please add YOLO-format .txt labels.")
        os.makedirs(RAW_LABEL_DIR, exist_ok=True)
        return []
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(RAW_IMAGE_DIR) if f.lower().endswith(valid_exts)])
    
    if not image_files:
        print(f"No images found in '{RAW_IMAGE_DIR}/'")
        return []
    
    print(f"\nFound {len(image_files)} images in '{RAW_IMAGE_DIR}/'")
    
    valid_pairs = []
    skipped_no_label = 0
    skipped_blurry = 0
    skipped_invalid = 0
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(RAW_IMAGE_DIR, img_file)
        label_path = os.path.join(RAW_LABEL_DIR, base_name + '.txt')
        
        # Check if label file exists
        if not os.path.exists(label_path):
            skipped_no_label += 1
            continue
        
        # Validate image (loadable + not overly blurry)
        is_valid, image = validate_image(img_path)
        if not is_valid:
            skipped_blurry += 1
            continue
        
        # Clean labels: remove invalid annotations
        valid_lines = clean_label_file(image, label_path, class_id=0)
        if not valid_lines:
            skipped_invalid += 1
            continue
        
        valid_pairs.append({
            'image_path': img_path,
            'label_lines': valid_lines,
            'filename': img_file
        })
    
    # Print cleaning summary
    print(f"\n--- Data Cleaning Summary ---")
    print(f"  Valid pairs:      {len(valid_pairs)}")
    print(f"  Skipped (no label):  {skipped_no_label}")
    print(f"  Skipped (blurry):    {skipped_blurry}")
    print(f"  Skipped (invalid):   {skipped_invalid}")
    
    return valid_pairs


def generate_synthetic_data(valid_pairs):
    """
    Generates augmented/synthetic dataset (Dataset 2) from valid image-label pairs.
    
    First copies valid images and labels to a temp directory, then
    applies augmentation to create synthetic variants.
    
    Args:
        valid_pairs: List of valid image-label pair dicts.
        
    Returns:
        list: Additional synthetic pairs to add to the dataset.
    """
    print(f"\n--- Generating Augmented Dataset (Dataset 2) ---")
    
    # Create temporary source directories for augmentation
    temp_img_dir = os.path.join(AUGMENTED_TEMP_DIR, "images")
    temp_lbl_dir = os.path.join(AUGMENTED_TEMP_DIR, "labels")
    aug_img_dir = os.path.join(AUGMENTED_TEMP_DIR, "aug_images")
    aug_lbl_dir = os.path.join(AUGMENTED_TEMP_DIR, "aug_labels")
    
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_lbl_dir, exist_ok=True)
    
    # Copy valid images and labels to temp source
    for pair in valid_pairs:
        shutil.copy(pair['image_path'], os.path.join(temp_img_dir, pair['filename']))
        base = os.path.splitext(pair['filename'])[0]
        lbl_path = os.path.join(temp_lbl_dir, base + '.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(pair['label_lines']) + '\n')
    
    # Generate augmented dataset
    count = generate_augmented_dataset(
        src_image_dir=temp_img_dir,
        src_label_dir=temp_lbl_dir,
        dst_image_dir=aug_img_dir,
        dst_label_dir=aug_lbl_dir,
        num_augmentations=NUM_AUGMENTATIONS
    )
    
    # Collect augmented pairs
    aug_pairs = []
    if os.path.exists(aug_img_dir):
        valid_exts = ('.jpg', '.jpeg', '.png')
        for img_file in os.listdir(aug_img_dir):
            if not img_file.lower().endswith(valid_exts):
                continue
            base = os.path.splitext(img_file)[0]
            lbl_file = base + '.txt'
            lbl_path = os.path.join(aug_lbl_dir, lbl_file)
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                
                aug_pairs.append({
                    'image_path': os.path.join(aug_img_dir, img_file),
                    'label_lines': lines,
                    'filename': img_file
                })
    
    print(f"  Augmented pairs created: {len(aug_pairs)}")
    return aug_pairs


def split_and_copy(all_pairs):
    """
    Splits dataset into train/val sets and copies files to the structured directory.
    
    Args:
        all_pairs: Combined list of valid real + augmented image-label pairs.
    """
    if not all_pairs:
        print("\nNo valid data to split. Please add images and labels first.")
        return
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    
    # Split into train and validation
    split_idx = int(len(all_pairs) * TRAIN_SPLIT)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    def copy_subset(pairs, split_name):
        """Copies image-label pairs to the appropriate split directory."""
        for pair in pairs:
            base = os.path.splitext(pair['filename'])[0]
            ext = os.path.splitext(pair['filename'])[1]
            
            # Copy image
            dst_img = os.path.join(DATASET_DIR, 'images', split_name, pair['filename'])
            shutil.copy(pair['image_path'], dst_img)
            
            # Write cleaned labels
            dst_lbl = os.path.join(DATASET_DIR, 'labels', split_name, base + '.txt')
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(pair['label_lines']) + '\n')
    
    copy_subset(train_pairs, 'train')
    copy_subset(val_pairs, 'val')
    
    print(f"\n--- Dataset Split Summary ---")
    print(f"  Total samples:     {len(all_pairs)}")
    print(f"  Training set:      {len(train_pairs)} ({TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation set:    {len(val_pairs)} ({(1-TRAIN_SPLIT)*100:.0f}%)")


def cleanup_temp():
    """Removes temporary augmentation directory."""
    if os.path.exists(AUGMENTED_TEMP_DIR):
        shutil.rmtree(AUGMENTED_TEMP_DIR)
        print(f"\n  Cleaned up temporary directory: {AUGMENTED_TEMP_DIR}/")


def main():
    """Main pipeline: clean → augment → split → copy."""
    print("=" * 60)
    print("  LICENSE PLATE DETECTION — Dataset Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Create output directory structure
    print("\n[Step 1] Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Collect and clean valid image-label pairs (Dataset 1: Real-world)
    print("\n[Step 2] Collecting and cleaning real-world data (Dataset 1)...")
    valid_pairs = collect_valid_pairs()
    
    # Step 3: Generate augmented/synthetic data (Dataset 2)
    print("\n[Step 3] Generating augmented/synthetic data (Dataset 2)...")
    if valid_pairs:
        aug_pairs = generate_synthetic_data(valid_pairs)
    else:
        aug_pairs = []
        print("  Skipped augmentation (no valid source data).")
    
    # Step 4: Combine and split into train/val
    print("\n[Step 4] Splitting dataset (80/20)...")
    all_pairs = valid_pairs + aug_pairs
    split_and_copy(all_pairs)
    
    # Step 5: Cleanup
    cleanup_temp()
    
    print("\n" + "=" * 60)
    print("  Dataset preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
