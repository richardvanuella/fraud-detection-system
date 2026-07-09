from pyexpat import features

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model & scaler
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'))
scaler_amount = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

app = FastAPI(title="Fraud Detection API", version="1.0")

# Definisi input
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = dict(transaction)
    
    # Scale Amount & Time
    amount_scaled = float(scaler_amount.transform([[data['Amount']]])[0][0])
    time_scaled = float(scaler_amount.transform([[data['Time']]])[0][0])
    
    # Susun fitur sesuai urutan training (V1-V28, Amount_scaled, Time_scaled)
    features = [data[f'V{i}'] for i in range(1, 29)]
    features.append(amount_scaled)
    features.append(time_scaled)
    
    features = np.array(features).reshape(1, -1)
    
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    print(f"Probability raw: {model.predict_proba(features)}")
    print(f"Features: {features}")
    return {
        "is_fraud": bool(prediction),
        "confidence": round(probability, 4),
        "status": "FRAUD DETECTED" if prediction == 1 else "NORMAL"
    }