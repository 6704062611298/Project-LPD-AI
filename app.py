"""
app.py - Main Streamlit Application Entry Point

License Plate Detection Web Application with 4 pages:
1. ML Model Explanation — dataset, features, algorithms, training
2. DL Model Explanation — YOLO architecture, training steps, pretrained model
3. ML Model Testing — upload image and get ML ensemble prediction
4. DL Model Testing — upload image and get YOLO detection with bounding boxes

Usage:
    streamlit run app.py
"""

import streamlit as st


# ===================== Page Configuration =====================
st.set_page_config(
    page_title="License Plate Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point with page navigation."""
    
    # Sidebar navigation
    st.sidebar.title("🚗 License Plate Detection")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "🏠 Home",
            "📊 ML Model Explanation",
            "🧠 DL Model Explanation",
            "🔬 ML Model Testing",
            "🎯 DL Model Testing"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Project Info**  
        License Plate Detection using  
        ML Ensemble & YOLOv8
        
        Built with ❤️ using Streamlit
        """
    )
    
    # Route to the selected page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 ML Model Explanation":
        from app.ml_explanation import show_ml_explanation
        show_ml_explanation()
    elif page == "🧠 DL Model Explanation":
        from app.dl_explanation import show_dl_explanation
        show_dl_explanation()
    elif page == "🔬 ML Model Testing":
        from app.ml_testing import show_ml_testing
        show_ml_testing()
    elif page == "🎯 DL Model Testing":
        from app.dl_testing import show_dl_testing
        show_dl_testing()


def show_home():
    """Displays the home page with project overview."""
    
    st.title("🚗 License Plate Detection System")
    st.markdown("### AI-Powered License Plate Detection using Machine Learning & Deep Learning")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            #### 🤖 Machine Learning Model
            - **Algorithms**: Random Forest, SVM, KNN
            - **Ensemble**: Voting Classifier (Soft Voting)
            - **Features**: Color Histogram, Edges, Texture (LBP), HOG
            - **Task**: Binary classification (Plate / No Plate)
            """
        )
    
    with col2:
        st.markdown(
            """
            #### 🧠 Deep Learning Model
            - **Architecture**: YOLOv8 (You Only Look Once v8)
            - **Pretrained**: COCO dataset (transfer learning)
            - **Task**: Object detection (bounding boxes)
            - **Class**: `license_plate`
            """
        )
    
    st.markdown("---")
    
    st.markdown(
        """
        #### 📁 Project Structure
        ```
        project/
        ├── dataset/              # Structured dataset (train/val splits)
        ├── models/               # Saved trained models
        ├── app/                  # Streamlit page modules
        ├── utils/                # Utility functions
        ├── output/               # Detection outputs & cropped plates
        ├── train_ml.py           # ML ensemble training script
        ├── train_dl.py           # YOLOv8 training script
        ├── prepare_dataset.py    # Data preparation pipeline
        ├── inference.py          # CLI inference script
        ├── app.py                # Streamlit web application
        ├── data.yaml             # YOLOv8 dataset config
        ├── requirements.txt      # Python dependencies
        └── README.md             # Project documentation
        ```
        """
    )
    
    st.markdown("---")
    
    st.markdown(
        """
        #### 🚀 Getting Started
        1. **Prepare Data**: Place raw images in `raw_images/` and labels in `raw_labels/`, then run `python prepare_dataset.py`
        2. **Train ML Model**: Run `python train_ml.py`
        3. **Train DL Model**: Run `python train_dl.py`
        4. **Run Inference**: Run `python inference.py path/to/image.jpg`
        5. **Launch Web App**: Run `streamlit run app.py`
        """
    )
    
    st.markdown("---")
    
    st.markdown(
        """
        #### 📚 References & Sources
        
        **Dataset**
        - 📦 [License Plate Recognition Dataset — Kaggle](https://www.kaggle.com/datasets) 
          — Real-world license plate images with bounding box annotations in YOLO format
        
        **Frameworks & Libraries**
        - 🔮 [Ultralytics YOLOv8](https://docs.ultralytics.com/) — Object detection framework
        - 🤖 [scikit-learn](https://scikit-learn.org/) — Machine learning library
        - 👁️ [OpenCV (cv2)](https://docs.opencv.org/) — Computer vision library
        - 🖼️ [Streamlit](https://streamlit.io/) — Web application framework
        
        **AI Assistance**
        - 🤖 [Claude by Anthropic](https://claude.ai/) — AI assistant used for code development, debugging, and project structuring
        
        > 📖 *See each model's Explanation page for detailed academic references.*
        """
    )


if __name__ == '__main__':
    main()
