"""
preprocessing.py - Data Cleaning & Validation Utilities

This module provides functions to validate and clean license plate data:
- Blur detection using Laplacian variance
- Plate bounding box validation (size, aspect ratio, contrast)
- Label file cleaning to remove invalid annotations
"""

import cv2
import numpy as np
import os


def is_blurry(image, threshold=100.0):
    """
    Detects if an image is blurry using the variance of the Laplacian.
    
    Args:
        image: Input image (BGR format).
        threshold: Variance threshold below which the image is considered blurry.
        
    Returns:
        bool: True if the image is blurry, False otherwise.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute Laplacian variance — low variance = blurry
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_valid_plate(image, bbox_norm, min_size=30, min_aspect=1.0, max_aspect=6.0, min_contrast=25.0):
    """
    Validates a license plate bounding box based on multiple criteria.
    
    Args:
        image: Full image (BGR format).
        bbox_norm: Normalized YOLO bbox [x_center, y_center, width, height].
        min_size: Minimum plate dimension in pixels (default: 30).
        min_aspect: Minimum width/height ratio.
        max_aspect: Maximum width/height ratio.
        min_contrast: Minimum standard deviation of pixel intensities.
        
    Returns:
        bool: True if the plate is valid, False otherwise.
    """
    h, w = image.shape[:2]
    x_c, y_c, bw, bh = bbox_norm
    
    # 1. Check minimum size (ignore plates smaller than 30x30 pixels)
    box_w_px = bw * w
    box_h_px = bh * h
    if box_w_px < min_size or box_h_px < min_size:
        return False
    
    # 2. Check aspect ratio (license plates are typically wider than tall)
    aspect_ratio = box_w_px / max(box_h_px, 1)
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        return False
    
    # 3. Crop the plate region for image-quality checks
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)
    
    # Clamp coordinates to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    plate_crop = image[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return False
    
    # 4. Check if the plate region is blurry
    if is_blurry(plate_crop, threshold=100.0):
        return False
    
    # 5. Check contrast (standard deviation of grayscale intensities)
    gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
    if gray_crop.std() < min_contrast:
        return False
    
    return True


def clean_label_file(image, label_path, class_id=0):
    """
    Reads a YOLO-format label file and filters out invalid annotations.
    
    Args:
        image: The corresponding image (BGR format).
        label_path: Path to the YOLO annotation .txt file.
        class_id: Expected class ID for license plates (default: 0).
        
    Returns:
        list: List of valid annotation lines (strings).
    """
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    valid_lines = []
    for line in lines:
        parts = line.strip().split()
        # YOLO format: class_id x_center y_center width height
        if len(parts) < 5:
            continue
        
        try:
            cls = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
        except ValueError:
            continue
        
        # Only keep annotations for the target class
        if cls != class_id:
            continue
        
        # Validate the bounding box
        if is_valid_plate(image, bbox):
            valid_lines.append(line.strip())
    
    return valid_lines


def validate_image(image_path):
    """
    Validates that an image file can be loaded and is not too blurry for use.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        tuple: (is_valid, image) where is_valid is bool and image is the loaded image or None.
    """
    if not os.path.exists(image_path):
        return False, None
    
    image = cv2.imread(image_path)
    if image is None:
        return False, None
    
    # Check if the entire image is too blurry
    if is_blurry(image, threshold=50.0):
        return False, None
    
    return True, image
