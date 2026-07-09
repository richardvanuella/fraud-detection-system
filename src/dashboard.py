import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Real-Time Fraud Detection Dashboard")
st.markdown("Sistem deteksi transaksi mencurigakan secara otomatis menggunakan Machine Learning (XGBoost)")
st.markdown("---")

# ── Load dataset sample buat simulasi ──
@st.cache_data
def load_sample_data():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv'))
    scaler_mean = df['Amount'].mean()
    scaler_std = df['Amount'].std()
    return df, scaler_mean, scaler_std

df, amount_mean, amount_std = load_sample_data()

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🏦 Simulasi Transaksi", "📊 Model Performance", "📋 Riwayat Transaksi"])

with tab1:
    st.markdown("### 🏦 Simulasi Transaksi Real-Time")
    st.markdown("Klik tombol di bawah untuk mensimulasikan transaksi yang masuk ke sistem bank.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Simulasi Transaksi Normal", use_container_width=True):
            # Ambil random transaksi normal dari dataset
            normal_df = df[df['Class'] == 0].sample(1)
            row = normal_df.iloc[0]
            st.session_state['last_transaction'] = row
            st.session_state['last_type'] = 'normal'

    with col2:
        if st.button("🚨 Simulasi Transaksi Fraud", use_container_width=True):
            # Ambil random transaksi fraud dari dataset
            fraud_df = df[df['Class'] == 1].sample(1)
            row = fraud_df.iloc[0]
            st.session_state['last_transaction'] = row
            st.session_state['last_type'] = 'fraud'

    if 'last_transaction' in st.session_state:
        row = st.session_state['last_transaction']

        # Tampilkan info transaksi (yang bisa dimengerti user awam)
        st.markdown("#### 📄 Detail Transaksi")
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("💰 Nominal", f"${row['Amount']:.2f}")
        info_col2.metric("⏱️ Waktu", f"{int(row['Time']//3600)} jam {int((row['Time']%3600)//60)} menit")
        info_col3.metric("🔖 ID Transaksi", f"TXN-{random.randint(100000, 999999)}")

        # Kirim ke API
        v_features = {f'V{i}': float(row[f'V{i}']) for i in range(1, 29)}
        payload = {**v_features, "Amount": float(row['Amount']), "Time": float(row['Time'])}

        with st.spinner("🔄 Menganalisis transaksi..."):
            time.sleep(0.8)  # biar kerasa "real-time"
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                result = response.json()

                st.markdown("#### 🎯 Hasil Analisis")
                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    if result['is_fraud']:
                        st.error("🚨 TRANSAKSI MENCURIGAKAN!")
                        st.markdown("Transaksi ini **diblokir** dan dilaporkan ke tim analis.")
                    else:
                        st.success("✅ TRANSAKSI AMAN")
                        st.markdown("Transaksi ini **disetujui** oleh sistem.")

                with res_col2:
                    confidence = result['confidence'] * 100
                    st.metric("Tingkat Kecurigaan", f"{confidence:.1f}%")
                    if confidence > 50:
                        st.warning("Tingkat kecurigaan tinggi!")
                    else:
                        st.info("Tingkat kecurigaan rendah.")

                with res_col3:
                    actual = "FRAUD" if row['Class'] == 1 else "NORMAL"
                    predicted = "FRAUD" if result['is_fraud'] else "NORMAL"
                    if actual == predicted:
                        st.metric("Akurasi Prediksi", "✅ Benar")
                    else:
                        st.metric("Akurasi Prediksi", "❌ Salah")

                # Simpan ke riwayat
                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                st.session_state['history'].append({
                    "ID": f"TXN-{random.randint(100000, 999999)}",
                    "Nominal": f"${row['Amount']:.2f}",
                    "Prediksi": predicted,
                    "Aktual": actual,
                    "Kecurigaan": f"{confidence:.1f}%",
                    "Status": "✅ Benar" if actual == predicted else "❌ Salah"
                })

            except Exception as e:
                st.error(f"❌ API tidak bisa dihubungi. Pastikan FastAPI jalan!")

with tab2:
    st.markdown("### 📊 Performa Model XGBoost")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "XGBoost")
    m2.metric("ROC-AUC", "0.9745")
    m3.metric("Fraud Recall", "86%", help="86% fraud berhasil terdeteksi")
    m4.metric("Fraud Precision", "68%", help="68% prediksi fraud ternyata benar")

    st.markdown("---")
    st.markdown("#### 🔢 Confusion Matrix")

    cm = np.array([[56800, 64], [14, 84]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_xlabel('Prediksi Model')
    ax.set_ylabel('Kondisi Aktual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    st.markdown("""
    **Cara baca:**
    - **56800** transaksi normal → diprediksi benar sebagai Normal ✅
    - **84** transaksi fraud → berhasil terdeteksi ✅  
    - **64** transaksi normal → salah diprediksi sebagai Fraud ⚠️
    - **14** transaksi fraud → lolos tidak terdeteksi ❌
    """)

with tab3:
    st.markdown("### 📋 Riwayat Transaksi")

    if 'history' in st.session_state and len(st.session_state['history']) > 0:
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df, use_container_width=True)

        total = len(history_df)
        benar = history_df[history_df['Status'] == '✅ Benar'].shape[0]
        st.metric("Akurasi Sesi Ini", f"{benar}/{total} ({benar/total*100:.0f}%)")
    else:
        st.info("Belum ada transaksi. Coba simulasikan transaksi di tab pertama!")