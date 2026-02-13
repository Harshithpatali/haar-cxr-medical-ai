#  Haar-CXR  
### Dual-Branch Spatial + Haar Wavelet Medical AI System for Pneumonia Detection  

🔗 **Live Demo:**  
https://haar-cxr-medical-ai-jrudfjradkr2zxk2wiu5pm.streamlit.app/

---

##  Project Overview

**Haar-CXR** is a production-structured medical AI system designed for high-accuracy pneumonia detection from chest X-ray images.

Unlike standard CNN classifiers, this system integrates:

-  Spatial Deep Learning (ResNet18)
-  Frequency-Domain Signal Processing (Haar Wavelet Transform)
-  Feature Fusion Architecture
-  Statistical Validation
-  Calibration & Uncertainty Estimation
-  Grad-CAM Explainability
-  MLflow Experiment Tracking
-  Docker Deployment
-  Streamlit Live Application

This project demonstrates complete ML lifecycle ownership — from modeling and validation to deployment and explainability.

---

# 🏗 System Architecture

```
## 🏗 System Architecture

```mermaid
flowchart LR

A([Chest X-ray Input])

subgraph Spatial Domain
B[ResNet18 Feature Extractor]
end

subgraph Frequency Domain
C[Haar Wavelet Transform]
D[Frequency CNN]
end

E[Feature Fusion]
F[Binary Classifier]
G[MC Dropout Uncertainty]
H[Grad-CAM + Energy Analysis]

A --> B
A --> C
C --> D
B --> E
D --> E
E --> F
F --> G
F --> H
```

```

---

#  Core Technical Highlights

## 1️⃣ Dual-Branch Learning
- Extracts spatial representations using pretrained ResNet18  
- Captures frequency-domain features via 2D Haar wavelet transform  
- Fuses spatial and spectral embeddings for richer representation  

## 2️⃣ Robust Validation Strategy
- Stratified 5-Fold Cross-Validation  
- ROC-AUC, Accuracy, Precision, Recall, F1-score  
- Sensitivity & Specificity  
- Statistical significance testing (independent t-test)  

## 3️⃣ Calibration & Reliability
- Expected Calibration Error (ECE)  
- Monte Carlo Dropout for predictive uncertainty  
- Confidence scoring mechanism  

## 4️⃣ Explainable AI
- Grad-CAM heatmap visualization  
- Wavelet band energy comparison  
- Confidence-aware predictions  

---

# 📊 Engineering Stack

## 🧠 Core ML Stack

<p align="left">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/PyWavelets-4B8BBE?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>

</p>

## ⚙️ MLOps & Deployment

<p align="left">

<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Virtualenv-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

</p>


---

# 📁 Production-Grade Project Structure

```
haar_cxr/
│
├── configs/
├── src/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   ├── uncertainty/
│   └── inference/
│
├── cross_validate.py
├── train.py
├── app.py
├── Dockerfile
└── README.md
```

---

# ▶️ How to Run

### Train with Cross Validation
```bash
python cross_validate.py
```

### Start MLflow
```bash
mlflow ui
```

### Launch Streamlit App
```bash
streamlit run app.py
```

### Docker Deployment
```bash
docker build -t haar-cxr .
docker run -p 8501:8501 haar-cxr
```

---

#  What This Project Demonstrates

- Advanced architecture design (multi-branch learning)  
- Signal processing integration into deep learning pipelines  
- Model calibration & uncertainty quantification  
- Explainable AI for medical compliance  
- Clean modular production structure  
- End-to-end ML lifecycle ownership  
- Deployment-ready ML system  

---

# ⚠ Disclaimer

This system is for research purposes only and is not intended for clinical diagnosis.

---

# 👨‍💻 Author

**Harshith Devraj**  
M.Sc. Applied Mathematics & Computing  
Machine Learning | Medical AI | Signal Processing  
