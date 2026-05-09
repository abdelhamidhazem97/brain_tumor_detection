# 🧠 Brain Tumor Detection using Deep Learning

A comprehensive AI project to classify MRI brain images into 4 categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training](#training)
5. [Running the Application](#running-the-application)
6. [Expected Results](#expected-results)
7. [Technologies Used](#technologies-used)

---

## ⚙️ System Requirements

- Python 3.8+
- NVIDIA GPU (Recommended for training)
- RAM: 8GB+ (16GB recommended)

---

## 🛠 Installation

### 1. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Dataset
Ensure the `brain_tumor_dataset/` directory exists and contains:
```
brain_tumor_dataset/
├── Training/
│   ├── glioma/       (1400 images)
│   ├── meningioma/   (1400 images)
│   ├── notumor/      (1400 images)
│   └── pituitary/    (1400 images)
└── Testing/
    ├── glioma/       (400 images)
    ├── meningioma/   (400 images)
    ├── notumor/      (400 images)
    └── pituitary/    (400 images)
```

---

## 📁 Project Structure

```
cancer-detection-ai/
├── brain_tumor_dataset/     # Training and testing dataset
├── models/                  # Trained models
│   └── brain_tumor_model.h5
├── templates/               # Web interface templates
│   └── index.html
├── brain_tumor_detection.ipynb  # Main notebook (Training & Analysis)
├── app.py                   # Flask web application
├── requirements.txt         # Required Python packages
└── README.md               # This file
```

---

## 🎓 Training

### Running the Notebook
```bash
jupyter notebook brain_tumor_detection.ipynb
```

The notebook includes:
1. **Importing Libraries** - All necessary modules
2. **Data Loading & Preprocessing** - Reading and normalizing images
3. **Exploratory Data Analysis (EDA)** - Charts and statistics
4. **Model Building** - Using TensorFlow/Keras with Transfer Learning (VGG16)
5. **Training** - With techniques to prevent overfitting
6. **Evaluation** - Confusion matrix and detailed classification report
7. **Model Saving** - For use in the web application

### Training Strategies:
- Uses **VGG16** as a base model (Transfer Learning)
- **Data Augmentation** to increase training data size
- **Early Stopping** to prevent overfitting
- **Learning Rate Scheduling** for optimal training convergence
- Image Size: 224×224 pixels

---

## 🌐 Running the Application

```bash
python app.py
```

Then open your browser at: [http://localhost:5000](http://localhost:5000)

### Interface Features:
- 🖱 Drag and drop images
- 📊 Confidence scores for each category
- 🎨 Modern and responsive design
- 🌙 Dark mode support

---

## 📊 Expected Results

| Metric | Expected Value |
|---------|-----------------|
| Accuracy | ~95%+ |
| Precision | ~94%+ |
| Recall | ~94%+ |
| F1-Score | ~94%+ |

---

## 🔧 Technologies Used

| Technology | Usage |
|---------|-----------|
| Python 3.x | Programming Language |
| TensorFlow / Keras | Model Building & Training |
| VGG16 | Transfer Learning |
| Flask | Web Interface |
| OpenCV | Image Processing |
| NumPy / Pandas | Data Manipulation |
| Matplotlib / Seaborn | Data Visualization |
| Scikit-learn | Evaluation Metrics |

---

## 📝 Notes

- This project is for educational and research purposes only.
- It should not be used as a substitute for professional medical diagnosis.
- Always consult a specialist for real medical diagnoses.

---

## 📜 License

This project is licensed for educational and research use.
