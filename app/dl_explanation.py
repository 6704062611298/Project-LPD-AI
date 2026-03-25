"""
dl_explanation.py - Deep Learning Model Explanation Page

Streamlit page that explains the YOLOv8 deep learning model:
- YOLO architecture overview
- Training steps and configuration
- Pretrained model explanation (transfer learning)
"""

import streamlit as st


def show_dl_explanation():
    """Renders the DL Model Explanation page."""
    
    st.title("🧠 Deep Learning Model Explanation")
    st.markdown("### YOLOv8 — License Plate Detection")
    
    st.markdown("---")
    
    # ==================== YOLO ARCHITECTURE ====================
    st.header("1. 🏗️ YOLO Architecture")
    
    st.markdown(
        """
        **YOLO (You Only Look Once)** is a state-of-the-art real-time object detection 
        architecture. Version 8 (YOLOv8) by Ultralytics brings significant improvements 
        in accuracy, speed, and ease of use.
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            #### Key Components
            
            1. **Backbone (CSPDarknet)**
                - Extracts hierarchical image features
                - Uses Cross-Stage Partial connections
                - Progressive downsampling (640→320→160→80→40→20)
            
            2. **Neck (PANet / FPN)**
                - Feature Pyramid Network for multi-scale detection
                - Path Aggregation Network for bottom-up feature fusion
                - Combines low-level (edges) & high-level (semantics) features
            
            3. **Detection Head**
                - Anchor-free detection (no predefined boxes)
                - Decoupled head for classification + regression
                - Outputs: bounding boxes, confidence scores, class probabilities
            """
        )
    
    with col2:
        st.markdown(
            """
            #### YOLOv8 Variants
            
            | Model | Parameters | Speed | Accuracy |
            |-------|-----------|-------|----------|
            | YOLOv8n (Nano) | 3.2M | ⚡ Fastest | Good |
            | YOLOv8s (Small) | 11.2M | Fast | Better |
            | YOLOv8m (Medium) | 25.9M | Moderate | High |
            | YOLOv8l (Large) | 43.7M | Slower | Higher |
            | YOLOv8x (Extra) | 68.2M | Slowest | Highest |
            
            **We use YOLOv8n (Nano)** for this project because:
            - Lightweight and suitable for deployment
            - Fast inference speed
            - Good accuracy for single-class detection
            - Lower resource requirements
            """
        )
    
    st.markdown(
        """
        #### Detection Pipeline
        ```
        Input Image (640×640) → Backbone → Neck (FPN+PAN) → Detection Head → NMS → Results
                                   ↓           ↓                ↓
                              Feature Maps  Multi-Scale     BBox + Conf + Class
                              at 3 scales   Fusion          Predictions
        ```
        """
    )
    
    st.markdown("---")
    
    # ==================== TRAINING STEPS ====================
    st.header("2. 📋 Training Steps")
    
    st.markdown(
        """
        #### Step 1: Dataset Preparation
        ```
        dataset/
        ├── images/
        │   ├── train/    # 80% of images
        │   └── val/      # 20% of images
        └── labels/
            ├── train/    # YOLO format: class_id x_center y_center width height
            └── val/
        ```
        
        Each label file (`.txt`) contains normalized bounding box coordinates:
        ```
        0 0.5234 0.4167 0.1875 0.0833
        │   │       │      │      │
        │   │       │      │      └── height (normalized)
        │   │       │      └── width (normalized)
        │   │       └── y_center (normalized)
        │   └── x_center (normalized)
        └── class_id (0 = license_plate)
        ```
        """
    )
    
    st.markdown(
        """
        #### Step 2: Model Configuration
        
        | Parameter | Value | Description |
        |-----------|-------|-------------|
        | Model | YOLOv8n | Nano variant (smallest/fastest) |
        | Epochs | 50 | Number of training iterations |
        | Image Size | 640×640 | Input resolution |
        | Batch Size | 16 | Images per training step |
        | Optimizer | AdamW | Adaptive learning rate optimizer |
        | Learning Rate | Auto | Automatically determined by YOLO |
        
        #### Step 3: Data Augmentation (During Training)
        
        YOLOv8 applies built-in augmentations during training:
        
        | Augmentation | Setting | Effect |
        |-------------|---------|--------|
        | Rotation | ±10° | Rotates images randomly |
        | HSV Value | 0.2 | Adjusts brightness ±20% |
        | Horizontal Flip | 50% | Mirrors images randomly |
        | Mosaic | Enabled | Combines 4 images into 1 |
        | MixUp | Enabled | Blends 2 images together |
        
        #### Step 4: Training Execution
        ```bash
        python train_dl.py
        ```
        
        #### Step 5: Model Evaluation
        
        After training, YOLOv8 automatically generates:
        - **mAP@50**: Mean Average Precision at IoU=0.5
        - **mAP@50:95**: mAP averaged across IoU thresholds
        - **Precision / Recall curves**
        - **Confusion matrix**
        - **Training loss curves** (box loss, classification loss, DFL loss)
        """
    )
    
    st.markdown("---")
    
    # ==================== PRETRAINED MODEL ====================
    st.header("3. 🔄 Pretrained Model (Transfer Learning)")
    
    st.markdown(
        """
        #### What is Transfer Learning?
        
        Instead of training from scratch (random weights), we start with a model 
        **pretrained on the COCO dataset** (Common Objects in Context).
        
        **COCO Dataset**:
        - 330K+ images with 80 object categories
        - Objects include: cars, people, animals, everyday items
        - Model has already learned general visual features
        
        #### Why Transfer Learning?
        
        | Aspect | From Scratch | Transfer Learning |
        |--------|-------------|-------------------|
        | Training data needed | Very large | Small (~100-1000 images) |
        | Training time | Days/Weeks | Hours |
        | Low-level features | Must learn edges, textures | Already learned |
        | Performance | Poor without large data | Good even with small data |
        
        #### How It Works in Our Project
        
        ```
        YOLOv8n (COCO pretrained)
              │
              ├── Backbone weights: FROZEN initially, then fine-tuned
              │   - Already knows: edges, textures, shapes, patterns
              │
              ├── Neck: Fine-tuned for our specific features
              │
              └── Detection Head: Retrained for our single class
                  - Class 0: license_plate
        ```
        
        **Process**:
        1. Load `yolov8n.pt` (pretrained weights from COCO)
        2. Replace the detection head for 1 class (license_plate)
        3. Fine-tune the entire network on our license plate dataset
        4. The backbone retains its ability to extract visual features
        5. The head learns to detect specifically license plates
        
        #### Model Output
        
        For each detection, the model outputs:
        - **Bounding box**: `[x1, y1, x2, y2]` pixel coordinates
        - **Confidence score**: 0.0 to 1.0 (probability of detection)
        - **Class prediction**: `license_plate` (only 1 class)
        """
    )
    
    st.success(
        "**Result**: The trained model (`models/yolo_best.pt`) can detect license plates "
        "in real-time with high accuracy, even with a small training dataset."
    )
    
    st.markdown("---")
    
    st.markdown(
        """
        #### 📚 References & Sources
        
        **Academic Papers**
        - 📄 Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *"You Only Look Once: Unified, Real-Time Object Detection"*, CVPR 2016 — [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
        - 📄 Jocher, G., Chaurasia, A., & Qiu, J. (2023). *"Ultralytics YOLOv8"* — [GitHub](https://github.com/ultralytics/ultralytics)
        - 📄 Pan, S.J. & Yang, Q. (2010). *"A Survey on Transfer Learning"*, IEEE TKDE — Transfer learning methodology
        
        **Frameworks & Datasets**
        - 🔮 [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/) — Model architecture, training configuration, and augmentation parameters
        - 🏋️ [COCO Dataset](https://cocodataset.org/) — Pretrained weights source (330K+ images, 80 categories)
        - 📦 [License Plate Dataset — Kaggle](https://www.kaggle.com/datasets) — Training data with YOLO-format annotations
        
        **AI Assistance**
        - 🤖 [Claude by Anthropic](https://claude.ai/) — Used for researching YOLO architecture details, transfer learning concepts, and structuring explanations
        """
    )
