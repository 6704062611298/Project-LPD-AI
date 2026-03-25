"""
ml_explanation.py - ML Model Explanation Page

Streamlit page that explains the Machine Learning ensemble model:
- Dataset description and data sources
- Feature engineering methodology
- Algorithms used (Random Forest, SVM, KNN)
- Ensemble approach (Voting Classifier)
- Training process and evaluation
"""

import streamlit as st


def show_ml_explanation():
    """Renders the ML Model Explanation page."""
    
    st.title("📊 Machine Learning Model Explanation")
    st.markdown("### Ensemble-Based License Plate Classification")
    
    st.markdown("---")
    
    # ==================== DATASET DESCRIPTION ====================
    st.header("1. 📁 Dataset Description")
    
    st.markdown(
        """
        This project uses **two datasets** for training:

        #### Dataset 1: Real-World License Plate Images (Unstructured)
        - **Source**: Collected from various public license plate datasets and real-world captures
        - **Format**: JPEG/PNG images with YOLO-format annotations (`.txt` files)
        - **Characteristics**:
            - Varying lighting conditions (day, night, shadows)
            - Different camera angles and distances
            - Multiple plate formats and sizes
            - **Imperfect data**: includes blurry images, occluded plates, noisy backgrounds
            - Some images may have missing or incorrect labels

        #### Dataset 2: Augmented/Synthetic Dataset (Generated)
        - **Source**: Generated from Dataset 1 using automated augmentation pipeline
        - **Augmentation techniques applied**:
            - 🔄 Random rotation (±15°)
            - 🔆 Brightness adjustment (0.6× to 1.4×)
            - 🌫️ Gaussian blur (kernel 3–7)
            - 📡 Gaussian noise injection
        - **Purpose**: Increases dataset diversity and model robustness
        - **Imperfections**: Intentionally introduces blur and noise to simulate real-world conditions

        #### Data Cleaning Applied
        | Filter | Criteria | Rationale |
        |--------|----------|-----------| 
        | Blur detection | Laplacian variance < 100 | Removes images too blurry for recognition |
        | Minimum size | < 30×30 pixels | Plates too small to extract features |
        | Aspect ratio | Outside 1.0–6.0 range | Removes invalid bounding boxes |
        | Contrast check | Std deviation < 25 | Ensures sufficient visual detail |
        """
    )
    
    st.markdown("---")
    
    # ==================== FEATURE ENGINEERING ====================
    st.header("2. 🔧 Feature Engineering")
    
    st.markdown(
        """
        For the ML model, we extract **numerical feature vectors** from image patches
        to convert visual data into a format suitable for traditional classifiers.
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            #### Color Histogram (96 features)
            - Computes histogram for each BGR channel (32 bins each)
            - Captures color distribution of the image patch
            - Normalized to ensure scale invariance
            
            #### Edge Features (5 features)
            - **Edge density**: Ratio of edge pixels (Canny detector)
            - **Gradient magnitude**: Mean and standard deviation
            - **Directional ratios**: Horizontal vs vertical edge proportion
            - Captures structural edges typical of license plates
            """
        )
    
    with col2:
        st.markdown(
            """
            #### Texture / LBP Features (26 features)
            - Simplified Local Binary Pattern (LBP) approach
            - Compares each pixel with 8 neighbors
            - Histogram of LBP values captures texture patterns
            - Useful for distinguishing plate textures from backgrounds
            
            #### HOG Features (variable)
            - Histogram of Oriented Gradients
            - Block size: 16×16, Cell size: 8×8, 9 orientation bins
            - Captures shape and gradient structure
            - Effective for structured objects like license plates
            """
        )
    
    st.info(
        "**Total feature vector**: All features are concatenated into a single vector "
        "and standardized using `StandardScaler` before feeding to classifiers."
    )
    
    st.markdown("---")
    
    # ==================== ALGORITHMS ====================
    st.header("3. 🤖 Algorithms Used")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Random Forest", "SVM", "KNN", "Ensemble"
    ])
    
    with tab1:
        st.markdown(
            """
            #### Random Forest Classifier
            
            - **Type**: Ensemble of decision trees
            - **Strength**: Handles high-dimensional features well, robust to overfitting
            - **Configuration**:
                - `n_estimators = 100` (100 trees)
                - `max_depth = 20` (limits tree depth)
                - `random_state = 42` (reproducibility)
            
            **How it works**: Builds multiple decision trees on random subsets of features
            and data, then aggregates predictions via majority vote. Each tree learns
            different patterns, making the forest robust.
            
            ```python
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            ```
            """
        )
    
    with tab2:
        st.markdown(
            """
            #### Support Vector Machine (SVM)
            
            - **Type**: Maximum-margin classifier with kernel trick
            - **Strength**: Effective in high-dimensional spaces, memory efficient
            - **Configuration**:
                - `kernel = 'rbf'` (Radial Basis Function)
                - `C = 1.0` (regularization parameter)
                - `gamma = 'scale'` (kernel coefficient)
                - `probability = True` (enables soft voting)
            
            **How it works**: Finds the optimal hyperplane that separates classes with
            maximum margin. The RBF kernel maps data to higher dimensions for non-linear separation.
            
            ```python
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            ```
            """
        )
    
    with tab3:
        st.markdown(
            """
            #### K-Nearest Neighbors (KNN)
            
            - **Type**: Instance-based (lazy) learner
            - **Strength**: Simple, no training phase, adapts to local data patterns
            - **Configuration**:
                - `n_neighbors = 5` (5 nearest neighbors)
                - `weights = 'distance'` (closer neighbors have more influence)
            
            **How it works**: For a new sample, finds the K closest training samples
            in feature space and predicts based on their majority class. Distance-weighted
            voting gives closer neighbors more influence.
            
            ```python
            KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
            ```
            """
        )
    
    with tab4:
        st.markdown(
            """
            #### Voting Classifier (Ensemble)
            
            - **Strategy**: Soft Voting
            - **Combination**: Averages predicted probabilities from all three models
            - **Final prediction**: Class with highest average probability wins
            
            **Why ensemble?**: Different algorithms have different strengths.
            Combining them reduces individual weaknesses and improves overall accuracy.
            
            | Model | Strength | Weakness |
            |-------|----------|----------|
            | Random Forest | Non-linear, robust | Can overfit on small data |
            | SVM | High-dim effective | Slow on large datasets |
            | KNN | Adapts locally | Sensitive to noise |
            
            ```python
            VotingClassifier(
                estimators=[
                    ('random_forest', rf_clf),
                    ('svm', svm_clf),
                    ('knn', knn_clf)
                ],
                voting='soft'
            )
            ```
            """
        )
    
    st.markdown("---")
    
    # ==================== TRAINING PROCESS ====================
    st.header("4. 📈 Training Process")
    
    st.markdown(
        """
        #### Pipeline Overview
        
        ```
        Raw Images → Data Cleaning → Feature Extraction → StandardScaler → Ensemble Training → Evaluation
        ```
        
        #### Step-by-Step Process
        
        1. **Data Loading**: 
           - Load labeled image patches from `dataset/images/train/` and `dataset/images/val/`
           - For each annotated plate region → crop as **positive sample** (label=1)
           - Generate random non-overlapping patches as **negative samples** (label=0)
        
        2. **Feature Extraction**:
           - Resize all patches to 128×128 pixels
           - Extract: Color Histogram + Edge + Texture + HOG features
           - Concatenate into single feature vector per sample
        
        3. **Preprocessing**:
           - Apply `StandardScaler` to normalize features (zero mean, unit variance)
        
        4. **Model Training**:
           - Train Random Forest, SVM, and KNN independently
           - Combine into `VotingClassifier` with soft voting
           - Uses `Pipeline` to chain scaler + ensemble
        
        5. **Evaluation**:
           - Report accuracy on training and validation sets
           - Generate classification report (precision, recall, F1-score)
        
        6. **Model Saving**:
           - Save trained pipeline to `models/ml_ensemble_model.pkl` using `joblib`
        
        #### Running Training
        ```bash
        python train_ml.py
        ```
        """
    )
    
    st.markdown("---")
    
    st.markdown(
        """
        #### 📚 References & Sources
        
        **Academic Papers**
        - 📄 Dalal, N. & Triggs, B. (2005). *"Histograms of Oriented Gradients for Human Detection"*, CVPR 2005 — HOG feature extraction methodology
        - 📄 Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *"Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP"*, IEEE TPAMI — LBP texture features
        - 📄 Breiman, L. (2001). *"Random Forests"*, Machine Learning, 45(1), 5–32 — Random Forest algorithm
        - 📄 Cortes, C. & Vapnik, V. (1995). *"Support-Vector Networks"*, Machine Learning, 20(3), 273–297 — SVM algorithm
        - 📄 Cover, T. & Hart, P. (1967). *"Nearest Neighbor Pattern Classification"*, IEEE Trans. Information Theory — KNN algorithm
        
        **Frameworks & Libraries**
        - 🤖 [scikit-learn Documentation](https://scikit-learn.org/stable/) — RandomForestClassifier, SVC, KNeighborsClassifier, VotingClassifier, Pipeline
        - 👁️ [OpenCV Documentation](https://docs.opencv.org/) — Image processing, Canny edge detection, HOG descriptor
        - 📦 [License Plate Dataset — Kaggle](https://www.kaggle.com/datasets) — Training data source
        
        **AI Assistance**
        - 🤖 [Claude by Anthropic](https://claude.ai/) — Used for researching ML algorithms, feature engineering techniques, and structuring explanations
        """
    )
