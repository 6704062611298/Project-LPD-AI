"""
inference.py - License Plate Detection Inference Script

Command-line interface for running license plate detection on images.

Supports:
- YOLOv8-based detection with bounding boxes + confidence scores
- Automatic cropping and saving of detected plates
- Optional OCR text extraction

Usage:
    python inference.py path/to/image.jpg
    python inference.py path/to/image.jpg --ocr
    python inference.py path/to/image.jpg --model models/yolo_best.pt
"""

import os
import sys
import argparse
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.inference_utils import (
    detect_license_plate,
    crop_and_save,
    draw_detections,
    ocr_plate
)


def main():
    """
    Main inference entry point.
    
    Parses command-line arguments and runs license plate detection
    on the specified image.
    """
    parser = argparse.ArgumentParser(
        description="License Plate Detection — Inference Script"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image for detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo_best.pt",
        help="Path to the YOLO model weights (default: models/yolo_best.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/cropped_plates",
        help="Directory to save cropped plates (default: output/cropped_plates)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR text extraction on detected plates"
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        default=True,
        help="Save visualization with bounding boxes (default: True)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  LICENSE PLATE DETECTION — Inference")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"\nError: Image not found at '{args.image_path}'")
        return
    
    # Run detection
    print(f"\n  Input image: {args.image_path}")
    print(f"  Model:       {args.model}")
    print(f"  Confidence:  {args.conf}")
    
    detections = detect_license_plate(
        args.image_path,
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    if not detections:
        print("\n  No license plates detected.")
        return
    
    print(f"\n  Detected {len(detections)} license plate(s):")
    for i, det in enumerate(detections):
        print(f"    [{i+1}] Confidence: {det['confidence']:.2f}  BBox: {det['bbox']}")
    
    # Load image for cropping and visualization
    image = cv2.imread(args.image_path)
    
    # Crop and save detected plates
    print(f"\n  Saving cropped plates to: {args.output}/")
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    crop_paths = crop_and_save(image, detections, args.output, base_name)
    
    # Optional OCR
    if args.ocr:
        print("\n  Running OCR on detected plates...")
        for crop_path in crop_paths:
            plate_img = cv2.imread(crop_path)
            if plate_img is not None:
                text = ocr_plate(plate_img)
                print(f"    {os.path.basename(crop_path)}: '{text}'")
    
    # Save visualization
    if args.save_viz:
        viz_dir = "output/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        annotated = draw_detections(image, detections)
        viz_path = os.path.join(viz_dir, os.path.basename(args.image_path))
        cv2.imwrite(viz_path, annotated)
        print(f"\n  Visualization saved to: {viz_path}")
    
    print("\n" + "=" * 60)
    print("  Inference complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
