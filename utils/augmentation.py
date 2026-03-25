"""
augmentation.py - Data Augmentation Utilities

This module provides image augmentation functions for generating
synthetic/augmented datasets:
- Random rotation
- Random brightness adjustment
- Random Gaussian blur
- Batch augmented dataset generation
"""

import cv2
import numpy as np
import os
import random
import shutil


def augment_rotation(image, angle_range=(-15, 15)):
    """
    Applies random rotation to an image.
    
    Args:
        image: Input image (BGR format).
        angle_range: Tuple (min_angle, max_angle) in degrees.
        
    Returns:
        numpy.ndarray: Rotated image.
    """
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute rotation matrix and apply affine transformation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                              borderMode=cv2.BORDER_REFLECT_101)
    return rotated


def augment_brightness(image, factor_range=(0.6, 1.4)):
    """
    Adjusts image brightness by a random factor.
    
    Args:
        image: Input image (BGR format).
        factor_range: Tuple (min_factor, max_factor). <1 = darker, >1 = brighter.
        
    Returns:
        numpy.ndarray: Brightness-adjusted image.
    """
    factor = random.uniform(factor_range[0], factor_range[1])
    
    # Convert to HSV to adjust the V (value/brightness) channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_blur(image, kernel_range=(3, 7)):
    """
    Applies random Gaussian blur to an image.
    
    Args:
        image: Input image (BGR format).
        kernel_range: Tuple (min_kernel, max_kernel) — must be odd numbers.
        
    Returns:
        numpy.ndarray: Blurred image.
    """
    # Ensure kernel size is odd
    k = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    return blurred


def augment_noise(image, noise_level=25):
    """
    Adds random Gaussian noise to an image.
    
    Args:
        image: Input image (BGR format).
        noise_level: Standard deviation of the noise.
        
    Returns:
        numpy.ndarray: Noisy image.
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_random_augmentation(image):
    """
    Applies a random combination of augmentations to a single image.
    
    Args:
        image: Input image (BGR format).
        
    Returns:
        numpy.ndarray: Augmented image.
    """
    augmented = image.copy()
    
    # Randomly apply each augmentation with 50% probability
    if random.random() > 0.5:
        augmented = augment_rotation(augmented)
    if random.random() > 0.5:
        augmented = augment_brightness(augmented)
    if random.random() > 0.5:
        augmented = augment_blur(augmented)
    if random.random() > 0.3:
        augmented = augment_noise(augmented, noise_level=15)
    
    return augmented


def generate_augmented_dataset(src_image_dir, src_label_dir, dst_image_dir, dst_label_dir, num_augmentations=3):
    """
    Generates an augmented (synthetic) dataset from source images and labels.
    
    For each source image, creates `num_augmentations` augmented copies.
    The corresponding label files are copied as-is (annotations remain the same
    since augmentations are global transforms that don't change bbox positions significantly).
    
    Args:
        src_image_dir: Path to source images directory.
        src_label_dir: Path to source labels directory.
        dst_image_dir: Path to destination augmented images directory.
        dst_label_dir: Path to destination augmented labels directory.
        num_augmentations: Number of augmented copies per image.
        
    Returns:
        int: Total number of augmented images generated.
    """
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(src_image_dir) if f.lower().endswith(valid_exts)]
    
    count = 0
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        label_file = base_name + '.txt'
        
        img_path = os.path.join(src_image_dir, img_file)
        label_path = os.path.join(src_label_dir, label_file)
        
        # Read source image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Check if a corresponding label file exists
        has_label = os.path.exists(label_path)
        
        for aug_idx in range(num_augmentations):
            # Apply random augmentations
            augmented = apply_random_augmentation(image)
            
            # Save augmented image
            aug_name = f"{base_name}_aug{aug_idx}{ext}"
            aug_img_path = os.path.join(dst_image_dir, aug_name)
            cv2.imwrite(aug_img_path, augmented)
            
            # Copy label file if it exists
            if has_label:
                aug_label_name = f"{base_name}_aug{aug_idx}.txt"
                aug_label_path = os.path.join(dst_label_dir, aug_label_name)
                shutil.copy(label_path, aug_label_path)
            
            count += 1
    
    print(f"Generated {count} augmented images in '{dst_image_dir}'")
    return count
