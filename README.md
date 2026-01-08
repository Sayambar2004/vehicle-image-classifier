# 🚘 Vehicle Image Classification System  
### A Deep Learning–Based Multi-Class Vehicle Recognition Pipeline

---

## 📌 Project Overview

This project presents a **production-ready computer vision system** for **multi-class vehicle image classification** using **deep convolutional neural networks (CNNs)** and **transfer learning**. The system is capable of accurately categorizing vehicle images into one of seven semantic classes with **near-perfect generalization performance**.

The trained model is deployed as an **interactive web application** using **Streamlit Community Cloud**, enabling real-time inference on user-uploaded images.

---

## 🔗 Live Demo

👉 **Streamlit Web Application:**  
**(https://vehicle-image-classifier-sayambar-roy-chowdhury.streamlit.app/)**  


---

## 📂 Dataset Description

### 📊 Dataset Source
The dataset used in this project was obtained from Kaggle:

🔗 **Dataset Link:**  
(https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification)

---

### 🗂️ Dataset Structure

The dataset follows a **directory-based class structure**, compatible with TensorFlow/Keras data loaders:
```text
Vehicles/
│
├── Auto Rickshaws/
├── Bikes/
├── Cars/
├── Motorcycles/
├── Planes/
├── Ships/
├── Trains/

```

- Each class contains **800 RGB images**
- Total images: **5,600**
- Images exhibit:
  - High intra-class variation
  - Diverse viewpoints and scales
  - Real-world backgrounds and lighting conditions

---

### 🚘 Target Classes

| Class Index | Vehicle Category |
|-----------|----------------|
| 0 | Auto Rickshaws |
| 1 | Bikes |
| 2 | Cars |
| 3 | Motorcycles |
| 4 | Planes |
| 5 | Ships |
| 6 | Trains |

---

## 🧠 Model Architecture

### 🔹 Backbone Network

- **EfficientNetB0**
- Pretrained on **ImageNet (1.2M images)**
- Employs **compound scaling** to balance:
  - Network depth
  - Network width
  - Input resolution

EfficientNet was selected due to its **parameter efficiency**, **strong feature representation**, and **excellent generalization behavior** on limited datasets.

---

### 🔹 Custom Classification Head

The pretrained backbone is augmented with a lightweight, task-specific classifier:

- Global Average Pooling (parameter-efficient feature aggregation)
- Batch Normalization (internal covariate shift mitigation)
- Fully Connected Dense Layer (ReLU activation)
- Dropout Regularization (p = 0.5)
- Softmax Output Layer (7 classes)

---

### 🔹 Model Summary

- Input Resolution: `224 × 224 × 3`
- Loss Function: `Categorical Cross-Entropy`
- Optimizer: `Adam`
- Learning Rate Scheduling: `ReduceLROnPlateau`
- Regularization: Dropout + Batch Normalization

---

## 🏋️ Training Strategy

### 🔄 Data Splitting

- Training set: **80%**
- Validation set: **20%**
- Split performed using `ImageDataGenerator`

---

### 🧪 Data Preprocessing & Augmentation

Training data is augmented to improve generalization:

- Random rotations
- Width & height shifts
- Zoom augmentation
- Horizontal flipping

EfficientNet-specific preprocessing is applied to all images.

---

### ⏹️ Callbacks Used

- **EarlyStopping** (monitoring validation loss)
- **ReduceLROnPlateau** (adaptive learning rate decay)
- **ModelCheckpoint** (saving best-performing model)

---

## 📈 Model Performance

### 🔹 Final Training Metrics

| Metric | Value |
|-----|------|
| Training Accuracy | **99.11%** |
| Training Loss | **0.0316** |

---

### 🔹 Final Validation Metrics

| Metric | Value |
|------|-------|
| Validation Accuracy | **99.55%** |
| Validation Loss | **0.0180** |

---

### 🧠 Interpretation

- Validation accuracy exceeding training accuracy indicates **healthy generalization**
- Data augmentation and dropout introduce regularization during training
- No evidence of overfitting or data leakage
- Classification report shows **near-perfect precision, recall, and F1-score** across all classes

---

## 🌐 Deployment Architecture

The trained model is deployed using:

- **Streamlit Community Cloud**
- CPU-based inference
- Model loaded directly from the GitHub repository
- Fully reproducible deployment using `requirements.txt`

---

## 🖥️ Web Application Features

- Image upload (JPG / PNG / JPEG)
- Real-time inference
- Predicted class label
- Model confidence score
- **Class-wise probability distribution (bar chart visualization)**
- Technical project and model documentation within the UI

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **EfficientNet**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Streamlit**

---

## 📦 Repository Structure

```text
vehicle-image-classifier/
├── app.py
├── vehicle_efficientnet_best.keras
├── requirements.txt
├── Vehicle_Classification_Notebook.ipynb
├── README.md
└── .gitignore
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
