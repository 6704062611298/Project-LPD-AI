"""
ml_testing.py - ML Model Testing Page

Streamlit page for testing the ML ensemble model:
- Upload image
- Extract features and run ML prediction
- Show classification result (Plate / No Plate)
- Display prediction probabilities
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image


def show_ml_testing():
    """Renders the ML Model Testing page."""
    
    st.title("🔬 ML Model Testing")
    st.markdown("### Test the Machine Learning Ensemble Model")
    st.markdown(
        "Upload an image patch to classify whether it contains a license plate "
        "using the trained ensemble model (Random Forest + SVM + KNN)."
    )
    
    st.markdown("---")
    
    # Check if model exists
    model_path = "models/ml_ensemble_model.pkl"
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.warning(
            "⚠️ **ML model not found!**  \n"
            f"Expected model at: `{model_path}`  \n"
            "Please run `python train_ml.py` first to train the ensemble model."
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="ml_upload"
    )
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to load image. Please try a different file.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, use_container_width=True)
            st.caption(f"Size: {image.shape[1]}×{image.shape[0]} pixels")
        
        with col2:
            st.subheader("🔍 Prediction Result")
            
            if not model_exists:
                st.error("Cannot predict — model not trained yet!")
                return
            
            # Run prediction
            with st.spinner("Extracting features and predicting..."):
                try:
                    from utils.inference_utils import predict_ml
                    
                    result = predict_ml(image, model_path=model_path)
                    
                    if result['label'] == -1:
                        st.error("Prediction error. Check model and dependencies.")
                        return
                    
                    # Display result with styling
                    if result['label'] == 1:
                        st.success(f"### ✅ {result['label_name']}")
                        st.balloons()
                    else:
                        st.info(f"### ❌ {result['label_name']}")
                    
                    # Show probabilities if available
                    if result['probabilities']:
                        st.markdown("#### Confidence Scores")
                        
                        prob_no_plate = result['probabilities'][0]
                        prob_plate = result['probabilities'][1]
                        
                        st.progress(prob_plate, text=f"License Plate: {prob_plate:.1%}")
                        st.progress(prob_no_plate, text=f"No Plate: {prob_no_plate:.1%}")
                    
                    # Show feature extraction details
                    with st.expander("🔧 Feature Extraction Details"):
                        from utils.feature_extraction import (
                            extract_color_histogram,
                            extract_edge_features,
                            extract_texture_features,
                            extract_hog_features
                        )
                        
                        color_f = extract_color_histogram(image)
                        edge_f = extract_edge_features(image)
                        texture_f = extract_texture_features(image)
                        hog_f = extract_hog_features(image)
                        
                        st.markdown(f"""
                        | Feature Type | Dimensions |
                        |-------------|-----------|
                        | Color Histogram | {len(color_f)} |
                        | Edge Features | {len(edge_f)} |
                        | Texture (LBP) | {len(texture_f)} |
                        | HOG Features | {len(hog_f)} |
                        | **Total** | **{len(color_f) + len(edge_f) + len(texture_f) + len(hog_f)}** |
                        """)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to test the ML model.")
