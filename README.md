# 🚗 License Plate Detection System

License Plate Detection system using **Machine Learning (Ensemble)** and **Deep Learning (YOLOv8)** with a **Streamlit** web application.

---

## 📁 Project Structure

```
project/
├── dataset/                  # Structured dataset (auto-generated)
│   ├── images/train/         # Training images (80%)
│   ├── images/val/           # Validation images (20%)
│   ├── labels/train/         # Training labels (YOLO format)
│   └── labels/val/           # Validation labels
├── models/                   # Saved trained models
│   ├── ml_ensemble_model.pkl # ML ensemble (RF + SVM + KNN)
│   └── yolo_best.pt          # YOLOv8 best weights
├── app/                      # Streamlit page modules
│   ├── ml_explanation.py     # ML model explanation page
│   ├── dl_explanation.py     # DL model explanation page
│   ├── ml_testing.py         # ML model testing page
│   └── dl_testing.py         # DL model testing page
├── utils/                    # Utility modules
│   ├── preprocessing.py      # Data cleaning & validation
│   ├── augmentation.py       # Data augmentation functions
│   ├── feature_extraction.py # Feature extraction for ML
│   └── inference_utils.py    # Detection & inference utilities
├── output/                   # Detection outputs
│   ├── cropped_plates/       # Cropped license plate images
│   └── visualizations/       # Annotated detection images
├── raw_images/               # Raw input images (user-provided)
├── raw_labels/               # Raw YOLO-format labels (user-provided)
├── prepare_dataset.py        # Data preparation pipeline
├── train_ml.py               # ML ensemble training script
├── train_dl.py               # YOLOv8 training script
├── inference.py              # CLI inference script
├── app.py                    # Streamlit web application
├── data.yaml                 # YOLOv8 dataset configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your raw license plate images and YOLO-format annotations:

```bash
raw_images/    # Put .jpg/.png images here
raw_labels/    # Put corresponding .txt labels here (YOLO format)
```

Then run the preparation pipeline:

```bash
python prepare_dataset.py
```

This will:
- Validate and clean data (remove blurry, small plates)
- Generate augmented/synthetic dataset
- Split into 80% train / 20% validation
- Create structured `dataset/` directory

### 3. Train Models

**ML Ensemble Model** (Random Forest + SVM + KNN):
```bash
python train_ml.py
```

**YOLOv8 Deep Learning Model**:
```bash
python train_dl.py
```

### 4. Run Inference

```bash
python inference.py path/to/image.jpg
python inference.py path/to/image.jpg --ocr    # With OCR text extraction
```

### 5. Launch Web Application

```bash
streamlit run app.py
```

---

## 🤖 Models

### Machine Learning Ensemble
- **Algorithms**: Random Forest, SVM, KNN
- **Ensemble Method**: Voting Classifier (Soft Voting)
- **Features**: Color Histogram, Edge Features, Texture (LBP), HOG
- **Task**: Binary classification — Plate vs. No Plate

### Deep Learning (YOLOv8)
- **Architecture**: YOLOv8n (Nano)
- **Pretrained on**: COCO dataset (transfer learning)
- **Training**: 50 epochs, image size 640×640
- **Task**: Object detection — License Plate bounding boxes
- **Class**: `0 = license_plate`

---

## 🌐 Web Application Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and getting started |
| 📊 ML Explanation | Dataset, features, algorithms, training process |
| 🧠 DL Explanation | YOLO architecture, training, transfer learning |
| 🔬 ML Testing | Upload image → ML ensemble prediction |
| 🎯 DL Testing | Upload image → YOLO detection with bounding boxes |

---

## 📊 Datasets

1. **Real-World Dataset**: License plate images with YOLO annotations
2. **Augmented Dataset**: Synthetically generated via rotation, brightness, blur, noise

### Data Cleaning
- Remove blurry images (Laplacian variance < 100)
- Ignore small plates (< 30×30 pixels)
- Filter invalid aspect ratios
- Check image contrast

---

## ☁️ Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `app.py` as the main file
5. Deploy!

---

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

---

## 📄 License

This project is developed for academic purposes.
