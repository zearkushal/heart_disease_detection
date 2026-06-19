from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your exported notebook assets
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'heart_disease_detection_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.joblib')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Extract Numeric Features
    age = float(request.form['age'])
    glucose = float(request.form['glucose_mg_dl'])
    cholesterol = float(request.form['cholesterol_mg_dl'])
    systolic_bp = float(request.form['systolic_bp'])
    diastolic_bp = float(request.form['diastolic_bp'])
    bmi = float(request.form['bmi'])
    heart_rate = float(request.form['heart_rate'])
    
    # 2. Extract and Map Binary Categorical Features (Text to 0/1)
    gender = 1 if request.form['gender'] == 'Male' else 0
    smoking = 1 if request.form['smoking'] == 'Yes' else 0
    alcohol = 1 if request.form['alcohol_consumption'] == 'Yes' else 0
    family_hist = 1 if request.form['family_history'] == 'Yes' else 0
    
    # 3. Extract and Map Ordinal Feature (Low: 0, Medium: 1, High: 2)
    activity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    physical_activity = activity_map[request.form['physical_activity']]
    
    # 4. Construct the complete 11-feature array 
    # CRITICAL: This array MUST match your exact notebook column order!
    raw_features = np.array([[
        age, gender, glucose, cholesterol, systolic_bp, 
        diastolic_bp, bmi, heart_rate, smoking, 
        alcohol, physical_activity, family_hist
    ]])
    
    # 5. Partial Scaling Trick
    # Remember: StandardScaler was only fitted on the numeric columns!
    # We pull the numeric values out, scale them, and put them back.
    numeric_indices = [0, 2, 3, 4, 5, 6, 7]  # Positions of age, glucose, cholesterol, etc.
    raw_features[:, numeric_indices] = scaler.transform(raw_features[:, numeric_indices])
    
    # 6. Run Inference
    prediction = model.predict(raw_features)
    probability = model.predict_proba(raw_features)[0][1] * 100
    
    if prediction[0] == 1:
        result = f"High Risk. Probability of Heart Disease: {probability:.2f}%"
    else:
        result = f"Low Risk. Probability of Heart Disease: {probability:.2f}%"
        
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)