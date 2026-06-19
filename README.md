# Sinlense-AI-
## Dataset

This project uses the HAM10000 (Human Against Machine with 10000 training images) dataset for skin disease classification.

Due to GitHub file size limits, the dataset is not included in this repository.

You can download the dataset from here:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# 🧠 SkinLens-AI: Hybrid Deep Learning System for Skin Lesion Analysis

## 📌 Overview

**SkinLens-AI** is an end-to-end intelligent diagnostic support system designed for automated skin lesion analysis using deep learning and clinical feature integration. The system combines convolutional neural network (CNN) predictions with dermatological ABCD rule-based features to generate a **hybrid risk score**.

This project demonstrates a real-world deployment pipeline involving:

* Image preprocessing
* Deep learning inference
* Feature engineering
* Hybrid decision modeling
* Automated report generation

---

## 🚀 Key Features

* 🔍 **Multiclass Classification**: Detects lesions as:

  * Benign
  * Suspicious
  * Pre-Cancerous
  * Cancerous

* 🧠 **Hybrid Risk Scoring Model**:
  Combines ML predictions with clinical features for improved reliability

* 📊 **ABCD Feature Analysis**:

  * A → Asymmetry
  * B → Border Irregularity
  * C → Color Variation
  * D → Diameter

* 📄 **Automated PDF Report Generation**

* 🌐 **Flask-based Web Interface**

* 📷 **Image Segmentation Support**

---

## 🏗️ System Architecture

```
User Input (Image)
        ↓
Preprocessing (Resize, Normalize)
        ↓
CNN Model (Inference)
        ↓
ABCD Feature Extraction
        ↓
Hybrid Risk Scoring
        ↓
Prediction + Confidence
        ↓
PDF Report Generation
```

---

## 🧮 Hybrid Risk Model

The system implements a fusion-based scoring mechanism:

Risk Score = 0.7 × Model Prediction + 0.3 × Clinical Features

Where:

* **Model Prediction** = Probability of malignant classes
* **Clinical Features** = Weighted ABCD metrics

Clinical score:

Clinical = 0.4A + 0.3B + 0.2C + 0.1D

This approach improves robustness by integrating **data-driven** and **domain-driven** insights.

---

## ⚙️ Tech Stack

* **Backend**: Flask
* **ML Framework**: TensorFlow / Keras
* **Image Processing**: OpenCV, NumPy
* **Report Generation**: FPDF
* **Frontend**: HTML (Jinja Templates)

---

## 📁 Project Structure

```
SkinLens-AI/
│
├── app.py                  # Main Flask application
├── preprocess.py          # Image preprocessing & feature extraction
├── model/
│   ├── skinlens_model.h5
│   └── label_classes.npy
│
├── static/uploads/        # Uploaded images & reports
├── templates/
│   ├── index.html
│   └── result.html
│
└── README.md
```

---

## ▶️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/shalinibr001/Sinlense-AI-
cd Sinlense-AI-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python app.py
```

### 4. Open in Browser

```
http://127.0.0.1:5000
```

---

## 📊 Model Details

* Input Size: 224 × 224 RGB images
* Architecture: CNN-based classifier
* Output: 4-class probability distribution

---

## 📄 Output

The system generates:

* Predicted class with confidence score
* ABCD feature values
* Hybrid risk score (0–1)
* Downloadable diagnostic PDF report

---

## ⚠️ Disclaimer

This system is intended for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**.

---

## 🔗 GitHub Repository

👉 https://github.com/shalinibr001/Sinlense-AI-

---

## 👩‍💻 Author

**Shalini B R**
AI/ML Enthusiast | System Design Learner

---

## 🌟 Future Work

* Integration with mobile-based image capture
* Deployment using cloud infrastructure (AWS/GCP)
* Explainable AI (Grad-CAM visualizations)
* Real-time telemedicine integration

---

## 🧠 Research Insight

This project reflects a broader system design principle:

> Combining **machine learning predictions** with **domain knowledge features** significantly improves decision reliability — a concept widely applicable in recommendation systems, healthcare AI, and intelligent assistants.

---

