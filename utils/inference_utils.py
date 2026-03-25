"""
inference_utils.py - Inference & Detection Utilities

This module provides the core inference functions for license plate detection:
- YOLOv8-based detection (bounding boxes + confidence)
- Plate cropping and saving
- Optional OCR text extraction using EasyOCR
"""

import os
import cv2
import numpy as np


# Default model and output paths
YOLO_MODEL_PATH = 'models/yolo_best.pt'
ML_MODEL_PATH = 'models/ml_ensemble_model.pkl'
OUTPUT_CROP_DIR = 'output/cropped_plates'


def detect_license_plate(image_path, model_path=None, conf_threshold=0.25):
    """
    Detects license plates in an image using the trained YOLOv8 model.
    
    Args:
        image_path: Path to the input image.
        model_path: Path to the YOLO model weights file (default: models/yolo_best.pt).
        conf_threshold: Minimum confidence threshold for detections.
        
    Returns:
        list: List of detection dicts with keys 'bbox', 'confidence', 'class_name'.
              Returns empty list if no detections or on error.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed. Run: pip install ultralytics")
        return []
    
    if model_path is None:
        model_path = YOLO_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please train the model first.")
        return []
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'.")
        return []
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold)[0]
    
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names.get(cls_id, f"class_{cls_id}")
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class_name': cls_name
        })
    
    return detections


def crop_and_save(image, detections, output_dir=None, base_name="plate"):
    """
    Crops detected license plate regions from the image and saves them.
    
    Args:
        image: Input image (BGR format, numpy array).
        detections: List of detection dicts from detect_license_plate().
        output_dir: Directory to save cropped plates (default: output/cropped_plates).
        base_name: Base filename prefix for saved crops.
        
    Returns:
        list: Paths to saved cropped plate images.
    """
    if output_dir is None:
        output_dir = OUTPUT_CROP_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Crop the plate region
        plate_crop = image[y1:y2, x1:x2]
        
        if plate_crop.size > 0:
            filename = f"{base_name}_{i}_conf{conf:.2f}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, plate_crop)
            saved_paths.append(save_path)
            print(f"  Saved cropped plate: {save_path}")
    
    return saved_paths


def draw_detections(image, detections):
    """
    Draws bounding boxes and labels on the image for visualization.
    
    Args:
        image: Input image (BGR format, numpy array).
        detections: List of detection dicts from detect_license_plate().
        
    Returns:
        numpy.ndarray: Image with drawn bounding boxes and labels.
    """
    annotated = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        label = f"LP {conf:.2f}"
        
        # Draw bounding box (green)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated


def ocr_plate(plate_image):
    """
    Extracts text from a cropped license plate image using EasyOCR.
    
    Args:
        plate_image: Cropped license plate image (BGR format, numpy array).
        
    Returns:
        str: Extracted text from the license plate, or empty string on failure.
    """
    try:
        import easyocr
    except ImportError:
        print("Warning: easyocr not installed. Run: pip install easyocr")
        return ""
    
    try:
        # Initialize EasyOCR reader (English for license plate text)
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Convert BGR to RGB for EasyOCR
        rgb_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        
        # Perform OCR
        results = reader.readtext(rgb_image)
        
        # Combine all detected text fragments
        texts = [result[1] for result in results]
        combined_text = ' '.join(texts).strip()
        
        return combined_text
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


def predict_ml(image, model_path=None):
    """
    Predicts whether an image patch contains a license plate using the ML ensemble model.
    
    Args:
        image: Input image patch (BGR format, numpy array).
        model_path: Path to the saved ML model (default: models/ml_ensemble_model.pkl).
        
    Returns:
        dict: Prediction result with 'label' (0 or 1), 'label_name', and 'probabilities'.
    """
    try:
        import joblib
        from utils.feature_extraction import extract_all_features
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        return {'label': -1, 'label_name': 'error', 'probabilities': []}
    
    if model_path is None:
        model_path = ML_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Error: ML model not found at '{model_path}'. Please train first.")
        return {'label': -1, 'label_name': 'error', 'probabilities': []}
    
    # Load the model
    model = joblib.load(model_path)
    
    # Extract features from the image
    features = extract_all_features(image).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    label_names = {0: 'No License Plate', 1: 'License Plate Detected'}
    
    # Get prediction probabilities if available
    try:
        probabilities = model.predict_proba(features)[0].tolist()
    except AttributeError:
        probabilities = []
    
    return {
        'label': int(prediction),
        'label_name': label_names.get(int(prediction), 'Unknown'),
        'probabilities': probabilities
    }
