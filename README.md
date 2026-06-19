# Heart Disease Risk Assessment App

A full-stack machine learning web application that predicts the probability of heart disease based on 11 clinical and lifestyle features. The project utilizes a **Logistic Regression** model trained in a Jupyter Notebook, serialized with `joblib`, and served via a **Flask** backend with a clean HTML5/CSS3 frontend.

---
### 🌐 Live Demo
**Explore the live application here:** <a href="https://heart-disease-detection-60e9.onrender.com/" target="_blank" rel="noopener noreferrer">Heart Disease Predictor Interface</a>

<a href="https://heart-disease-detection-60e9.onrender.com/" target="_blank" rel="noopener noreferrer">![Deployment Status](https://img.shields.io/badge/Deployment-Live%20on%20Render-brightgreen?style=for-the-badge&logo=render)</a>

---
## 📊 Model Evaluation & Performance

The model was validated on a test dataset split, showing strong predictive power with a highly balanced precision and recall tradeoff. 

### Classification Report
* **Overall Accuracy:** `86%`

| Class | Condition | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Low/No Risk | 0.86 | 0.89 | 0.87 | 105 |
| **1** | High Risk | 0.87 | 0.84 | 0.86 | 95 |

### Key Metrics Broken Down:
* **Precision (87% for High Risk):** When the model alerts that a patient has a high risk of heart disease, it is correct 87% of the time. This minimizes false positives.
* **Recall (84% for High Risk):** The model successfully catches 84% of all actual heart disease cases in the dataset, which is critical for medical screening applications to avoid missing high-risk individuals.

---
## 🚀 Features
* **11-Feature ML Pipeline:** Considers key clinical metrics (Blood Pressure, Glucose, Cholesterol, BMI) along with demographic and lifestyle factors (Age, Smoking, Physical Activity).
* **Smart Hybrid Scaling:** Automatically isolates and scales continuous numerical variables using a pre-trained `StandardScaler` while preserving binary and categorical encodings.
* **Real-time Probability Estimation:** Displays the exact percentage risk alongside a clear classification (High Risk vs. Low Risk).
* **Lightweight Architecture:** Simple, responsive frontend UI making clean synchronous POST requests to a secure Flask API.

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
