"""
dl_testing.py - DL Model Testing Page

Streamlit page for testing the YOLOv8 deep learning model:
- Upload image
- Run YOLOv8 inference
- Display bounding boxes on the image
- Show cropped plates and detection details
- Optional OCR text extraction
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image


def show_dl_testing():
    """Renders the DL Model Testing page."""
    
    st.title("🎯 DL Model Testing")
    st.markdown("### Test the YOLOv8 License Plate Detector")
    st.markdown(
        "Upload an image to detect license plates using the trained YOLOv8 model. "
        "The model will draw bounding boxes around detected plates and crop them."
    )
    
    st.markdown("---")
    
    # Check if model exists
    model_path = "models/yolo_best.pt"
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.warning(
            "⚠️ **YOLO model not found!**  \n"
            f"Expected model at: `{model_path}`  \n"
            "Please run `python train_dl.py` first to train the YOLOv8 model."
        )
    
    # Settings sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for a detection to be shown"
        )
        enable_ocr = st.checkbox(
            "Enable OCR (Extract plate text)",
            value=False,
            help="Uses EasyOCR to extract text from detected plates"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="dl_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily for YOLO inference
        temp_dir = "output/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Read image for display
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to load image. Please try a different file.")
            return
        
        st.subheader("📷 Uploaded Image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True)
        st.caption(f"Size: {image.shape[1]}×{image.shape[0]} pixels")
        
        st.markdown("---")
        
        if not model_exists:
            st.error("Cannot run detection — model not trained yet!")
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return
        
        # Run YOLOv8 detection
        st.subheader("🔍 Detection Results")
        
        with st.spinner("Running YOLOv8 inference..."):
            try:
                from utils.inference_utils import (
                    detect_license_plate,
                    draw_detections,
                    crop_and_save,
                    ocr_plate
                )
                
                # Detect license plates
                detections = detect_license_plate(
                    temp_path,
                    model_path=model_path,
                    conf_threshold=conf_threshold
                )
                
                if not detections:
                    st.warning("No license plates detected in this image.")
                    st.info(
                        "💡 **Tips**: Try lowering the confidence threshold, "
                        "or ensure the image contains visible license plates."
                    )
                else:
                    # Display annotated image with bounding boxes
                    annotated = draw_detections(image, detections)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_rgb, caption="Detected License Plates", 
                             use_container_width=True)
                    
                    # Detection summary
                    st.success(f"**Detected {len(detections)} license plate(s)!**")
                    
                    # Show details for each detection
                    st.markdown("#### 📋 Detection Details")
                    
                    cols = st.columns(min(len(detections), 3))
                    
                    for i, det in enumerate(detections):
                        col_idx = i % 3
                        with cols[col_idx]:
                            x1, y1, x2, y2 = det['bbox']
                            conf = det['confidence']
                            
                            st.markdown(f"**Plate #{i+1}**")
                            
                            # Crop and display the plate
                            plate_crop = image[y1:y2, x1:x2]
                            if plate_crop.size > 0:
                                plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                st.image(plate_rgb, caption=f"Confidence: {conf:.1%}",
                                         use_container_width=True)
                                
                                # OCR if enabled
                                if enable_ocr:
                                    with st.spinner("Running OCR..."):
                                        text = ocr_plate(plate_crop)
                                        if text:
                                            st.info(f"📝 OCR Text: **{text}**")
                                        else:
                                            st.warning("Could not extract text")
                            
                            st.markdown(
                                f"- BBox: `[{x1}, {y1}, {x2}, {y2}]`\n"
                                f"- Size: {x2-x1}×{y2-y1} px\n"
                                f"- Confidence: {conf:.2%}"
                            )
                    
                    # Save cropped plates
                    crop_dir = "output/cropped_plates"
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    saved_paths = crop_and_save(image, detections, crop_dir, base_name)
                    
                    if saved_paths:
                        with st.expander("📁 Saved Cropped Plates"):
                            for path in saved_paths:
                                st.text(f"✓ {path}")
            
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    else:
        st.info("👆 Please upload an image to test the YOLO model.")
