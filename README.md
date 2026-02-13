# Haar-CXR  
### Dual-Branch Spatial + Haar Wavelet Medical AI System for Pneumonia Detection  

<p align="left">
<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-LiveApp-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-ExperimentTracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
</p>

🔗 **Live Application:**  
https://haar-cxr-medical-ai-jrudfjradkr2zxk2wiu5pm.streamlit.app/

---

#  Project Overview

**Haar-CXR** is a production-structured medical AI system for pneumonia detection from chest X-ray images.

Unlike standard CNN classifiers, Haar-CXR integrates:

-  Spatial Deep Learning (ResNet18)
-  Frequency-Domain Signal Processing (Haar Wavelet Transform)
-  Feature Fusion Architecture
-  Statistical Validation
-  Calibration & Uncertainty Estimation
-  Explainable AI (Grad-CAM)
-  MLflow-based experiment tracking
-  Docker containerization
-  Streamlit deployment

This project demonstrates full ML lifecycle ownership — from research modeling to production deployment.

---

# 🏗 System Architecture

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

---

# 🧠 Core Technical Highlights

## 🔹 Dual-Branch Learning
- Spatial representation via pretrained ResNet18  
- Frequency modeling via 2D Haar wavelet transform  
- Feature-level fusion of spatial and spectral embeddings  

## 🔹 Robust Validation
- Stratified 5-Fold Cross-Validation  
- ROC-AUC, Accuracy, Precision, Recall, F1-score  
- Sensitivity & Specificity  
- Statistical significance testing (t-test on wavelet energy)  

## 🔹 Calibration & Reliability
- Expected Calibration Error (ECE)  
- Monte Carlo Dropout for predictive uncertainty  
- Confidence-aware probability outputs  

## 🔹 Explainable AI
- Grad-CAM heatmaps for localized pneumonia regions  
- Wavelet band energy analysis  
- Uncertainty-aware predictions  

---

# 🧠 Core ML Stack

<p align="left">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/PyWavelets-4B8BBE?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
</p>

---

# ⚙️ MLOps & Deployment

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

### Start MLflow UI
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

# 💡 What This Project Demonstrates

- Advanced multi-branch architecture design  
- Integration of signal processing into deep learning pipelines  
- Model calibration & uncertainty quantification  
- Explainable AI for medical compliance  
- Modular, production-ready ML code structure  
- End-to-end ML lifecycle ownership  
- Deployment-ready containerized system  

---

# ⚠ Disclaimer

This system is for research purposes only and is not intended for clinical diagnosis.

---

# 👨‍💻 Author

**Harshith Devraj**  
M.Sc. Applied Mathematics & Computing  
Machine Learning | Medical AI | Signal Processing  
