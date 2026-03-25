"""
feature_extraction.py - Image Feature Extraction for ML Models

This module extracts numerical features from images for use with
traditional machine learning classifiers (Random Forest, SVM, KNN).

Features include:
- Color histogram (3-channel, flattened)
- Edge features (Canny edge density)
- Texture features (Local Binary Pattern approximation)
- HOG-like gradient features
"""

import cv2
import numpy as np


# Standard size for feature extraction (all images resized to this)
FEATURE_IMG_SIZE = (128, 128)


def extract_color_histogram(image, bins=32):
    """
    Extracts a color histogram feature vector from an image.
    
    Computes histograms for each BGR channel independently and
    concatenates them into a single flattened feature vector.
    
    Args:
        image: Input image (BGR format).
        bins: Number of histogram bins per channel.
        
    Returns:
        numpy.ndarray: Flattened color histogram feature vector (length = bins * 3).
    """
    # Resize to standard size for consistent feature dimensions
    resized = cv2.resize(image, FEATURE_IMG_SIZE)
    
    features = []
    for channel in range(3):  # B, G, R channels
        hist = cv2.calcHist([resized], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    return np.array(features, dtype=np.float32)


def extract_edge_features(image):
    """
    Extracts edge-based features using Canny edge detection.
    
    Computes:
    - Edge density (ratio of edge pixels to total pixels)
    - Mean and std of gradient magnitudes
    - Horizontal and vertical edge ratios
    
    Args:
        image: Input image (BGR format).
        
    Returns:
        numpy.ndarray: Edge feature vector (length = 5).
    """
    resized = cv2.resize(image, FEATURE_IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Sobel gradients for directional edge info
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mean = np.mean(gradient_magnitude)
    grad_std = np.std(gradient_magnitude)
    
    # Horizontal vs vertical edge ratio
    h_edges = np.sum(np.abs(sobel_x) > np.abs(sobel_y))
    v_edges = np.sum(np.abs(sobel_y) > np.abs(sobel_x))
    total = max(h_edges + v_edges, 1)
    h_ratio = h_edges / total
    v_ratio = v_edges / total
    
    return np.array([edge_density, grad_mean, grad_std, h_ratio, v_ratio], dtype=np.float32)


def extract_texture_features(image, num_points=8, radius=1):
    """
    Extracts texture features using a simplified Local Binary Pattern (LBP) approach.
    
    Computes a histogram of LBP-like patterns to capture texture information.
    
    Args:
        image: Input image (BGR format).
        num_points: Number of sampling points for LBP (used for histogram bins).
        radius: Radius of the LBP pattern (not directly used in simplified version).
        
    Returns:
        numpy.ndarray: Texture feature vector (length = 26).
    """
    resized = cv2.resize(image, FEATURE_IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Simplified LBP: compare each pixel to its neighbors
    # Use shifts to compare center pixel with surrounding pixels
    lbp_map = np.zeros_like(gray, dtype=np.uint8)
    
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
              (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for idx, (dy, dx) in enumerate(shifts):
        # Shift the grayscale image and compare
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp_map += ((gray >= shifted).astype(np.uint8) << idx)
    
    # Compute histogram of LBP values (uniform pattern approximation)
    lbp_hist = cv2.calcHist([lbp_map], [0], None, [26], [0, 256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
    
    return lbp_hist.astype(np.float32)


def extract_hog_features(image):
    """
    Extracts HOG (Histogram of Oriented Gradients) features.
    
    HOG captures shape and gradient structure, making it useful for
    detecting structured objects like license plates.
    
    Args:
        image: Input image (BGR format).
        
    Returns:
        numpy.ndarray: HOG feature vector.
    """
    resized = cv2.resize(image, FEATURE_IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Create HOG descriptor with specific parameters
    win_size = FEATURE_IMG_SIZE
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(gray)
    
    return features.flatten().astype(np.float32)


def extract_all_features(image):
    """
    Extracts and concatenates all feature types into a single feature vector.
    
    Combines: color histogram + edge features + texture features + HOG features.
    
    Args:
        image: Input image (BGR format).
        
    Returns:
        numpy.ndarray: Combined feature vector.
    """
    color_feats = extract_color_histogram(image)      # 96 features
    edge_feats = extract_edge_features(image)          # 5 features
    texture_feats = extract_texture_features(image)    # 26 features
    hog_feats = extract_hog_features(image)            # variable (depends on params)
    
    # Concatenate all features into one vector
    combined = np.concatenate([color_feats, edge_feats, texture_feats, hog_feats])
    return combined
