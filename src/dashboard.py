import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Real-Time Fraud Detection Dashboard")
st.markdown("---")

# ── Sidebar ──
st.sidebar.header("⚙️ Input Transaksi")
st.sidebar.markdown("Masukkan data transaksi untuk dicek:")

amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=149.62)
time = st.sidebar.number_input("Time (seconds)", min_value=0.0, value=0.0)

st.sidebar.markdown("**Feature V1 - V28** (hasil PCA, biarkan default untuk demo):")
v_features = {}
for i in range(1, 29):
    v_features[f'V{i}'] = st.sidebar.number_input(f"V{i}", value=round(np.random.uniform(-2, 2), 4), format="%.4f")

# ── Predict Button ──
if st.sidebar.button("🔎 Cek Transaksi", use_container_width=True):
    payload = {**v_features, "Amount": amount, "Time": time}
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        st.markdown("## 📊 Hasil Prediksi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result['is_fraud']:
                st.error("🚨 FRAUD DETECTED!")
            else:
                st.success("✅ NORMAL")
        
        with col2:
            st.metric("Confidence", f"{result['confidence']*100:.2f}%")
        
        with col3:
            st.metric("Status", result['status'])
            
    except Exception as e:
        st.error(f"❌ API tidak bisa dihubungi. Pastikan FastAPI jalan! Error: {e}")

st.markdown("---")

# ── Stats Section ──
st.markdown("## 📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "XGBoost")
col2.metric("ROC-AUC", "0.9745")
col3.metric("Fraud Recall", "86%")
col4.metric("Fraud Precision", "68%")

st.markdown("---")

# ── Confusion Matrix ──
st.markdown("## 🔢 Confusion Matrix (Test Set)")

cm = np.array([[56800, 64], [14, 84]])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)