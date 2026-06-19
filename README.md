# Heart Disease Risk Assessment App

A full-stack machine learning web application that predicts the probability of heart disease based on 11 clinical and lifestyle features. The project utilizes a **Logistic Regression** model trained in a Jupyter Notebook, serialized with `joblib`, and served via a **Flask** backend with a clean HTML5/CSS3 frontend.

---
### 🌐 Live Demo
**Explore the live application here:** [Heart Disease Predictor Interface](https://heart-disease-detection-60e9.onrender.com/)

[![Deployment Status](https://img.shields.io/badge/Deployment-Live%20on%20Render-brightgreen?style=for-the-badge&logo=render)](https://heart-disease-detection-60e9.onrender.com/)

---
## 🚀 Features
* **11-Feature ML Pipeline:** Considers key clinical metrics (Blood Pressure, Glucose, Cholesterol, BMI) along with demographic and lifestyle factors (Age, Smoking, Physical Activity).
* **Smart Hybrid Scaling:** Automatically isolates and scales continuous numerical variables using a pre-trained `StandardScaler` while preserving binary and categorical encodings.
* **Real-time Probability Estimation:** Displays the exact percentage risk alongside a clear classification (High Risk vs. Low Risk).
* **Lightweight Architecture:** Simple, responsive frontend UI making clean asynchronous/synchronous POST requests to a secure Flask api.

---

## 📁 Project Structure

```text
heart_disease_detection/
│
├── .venv/                   # Python Virtual Environment (Local only)
├── backend/
│   ├── static/              # CSS/Static assets
│   ├── templates/
│   │   └── index.html       # Web Form Interface
│   └── app.py               # Flask Web Server & Inference Pipeline
│
├── models/
│   ├── heart_disease_detection_model.joblib  # Trained Logistic Regression Model
│   └── scaler.joblib                        # Trained StandardScaler Instance
│
├── .gitignore               # Prevents tracking of venv and caches
├── README.md                # Project Documentation
└── requirements.txt         # Production dependencies
