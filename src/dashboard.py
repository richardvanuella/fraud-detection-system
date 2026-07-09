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
    page_icon="",
    layout="wide"
)

st.title("Real-Time Fraud Detection Dashboard")
st.markdown("Sistem deteksi transaksi mencurigakan secara otomatis menggunakan Machine Learning (XGBoost)")
st.markdown("---")

@st.cache_data
def load_sample_data():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv'))
    return df

df = load_sample_data()

# ── Mapping merchant ke pola transaksi ──
MERCHANT_PROFILES = {
    "Indomaret / Alfamart": {"amount_range": (10000, 200000), "fraud_risk": "low"},
    "Shopee / Tokopedia": {"amount_range": (50000, 5000000), "fraud_risk": "medium"},
    "Amazon / eBay (Luar Negeri)": {"amount_range": (100000, 10000000), "fraud_risk": "high"},
    "ATM Withdrawal": {"amount_range": (100000, 2000000), "fraud_risk": "medium"},
    "Transfer Bank": {"amount_range": (50000, 50000000), "fraud_risk": "medium"},
    "Restoran / Kafe": {"amount_range": (20000, 500000), "fraud_risk": "low"},
    "Merchant Tidak Dikenal": {"amount_range": (100000, 10000000), "fraud_risk": "high"},
}

tab1, tab2, tab3 = st.tabs(["Simulasi Transaksi", "Model Performance", "Riwayat Transaksi"])

with tab1:
    st.markdown("### Simulasi Transaksi")
    st.markdown("Isi form di bawah seperti transaksi kartu kredit biasa.")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Detail Transaksi")

        merchant = st.selectbox("Nama Merchant", list(MERCHANT_PROFILES.keys()))
        amount = st.number_input(
            "Nominal Transaksi (Rp)",
            min_value=1000,
            max_value=100000000,
            value=150000,
            step=10000
        )
        transaction_type = st.selectbox("Tipe Transaksi", ["Online", "Tap / Gesek Kartu", "Transfer", "ATM"])
        location = st.selectbox("Lokasi Transaksi", ["Dalam Negeri", "Luar Negeri"])
        hour = st.slider("Jam Transaksi", 0, 23, 14)

        is_overseas = location == "Luar Negeri"
        is_night = hour < 6 or hour > 22
        is_high_risk_merchant = MERCHANT_PROFILES[merchant]["fraud_risk"] == "high"

        # Konversi input ke fitur model
        # Ambil sample dari dataset sesuai risk level
        if is_overseas or is_high_risk_merchant:
            sample_pool = df[df['Class'] == 1] if (is_overseas and is_night) else df[df['Class'] == 0].sample(frac=0.01)
        else:
            sample_pool = df[df['Class'] == 0].sample(frac=0.01)

        base_row = sample_pool.sample(1).iloc[0]

        # Modifikasi V17 dan V14 berdasarkan risk (fitur paling berpengaruh)
        v_features = {f'V{i}': float(base_row[f'V{i}']) for i in range(1, 29)}
        if is_overseas and is_night:
            v_features['V17'] = random.uniform(-8, -5)
            v_features['V14'] = random.uniform(-7, -4)
            v_features['V12'] = random.uniform(-5, -3)
        elif is_overseas or is_high_risk_merchant:
            v_features['V17'] = random.uniform(-4, -2)
            v_features['V14'] = random.uniform(-3, -1)

        amount_normalized = amount / 10000

        if st.button("Proses Transaksi", use_container_width=True):
            st.session_state['pending_transaction'] = {
                "v_features": v_features,
                "amount": amount_normalized,
                "merchant": merchant,
                "amount_display": amount,
                "type": transaction_type,
                "location": location,
                "hour": hour
            }

    with col_result:
        st.markdown("#### Hasil Analisis")

        if 'pending_transaction' in st.session_state:
            txn = st.session_state['pending_transaction']

            with st.spinner("Menganalisis transaksi..."):
                time.sleep(0.8)
                payload = {**txn['v_features'], "Amount": txn['amount'], "Time": float(txn['hour'] * 3600)}

                try:
                    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                    result = response.json()

                    confidence = result['confidence'] * 100

                    # Info transaksi
                    st.markdown(f"**ID Transaksi:** TXN-{random.randint(100000, 999999)}")
                    st.markdown(f"**Merchant:** {txn['merchant']}")
                    st.markdown(f"**Nominal:** Rp {txn['amount_display']:,}")
                    st.markdown(f"**Tipe:** {txn['type']}")
                    st.markdown(f"**Lokasi:** {txn['location']}")
                    st.markdown(f"**Jam:** {txn['hour']:02d}:00")
                    st.markdown("---")

                    if result['is_fraud']:
                        st.error("TRANSAKSI DIBLOKIR")
                        st.markdown("Sistem mendeteksi pola mencurigakan pada transaksi ini. Tim analis akan segera menghubungi pemegang kartu.")
                    else:
                        st.success("TRANSAKSI DISETUJUI")
                        st.markdown("Transaksi berhasil diproses.")

                    st.metric("Tingkat Kecurigaan", f"{confidence:.1f}%")

                    # Faktor risiko
                    st.markdown("**Faktor yang dianalisis:**")
                    factors = []
                    if txn['location'] == "Luar Negeri":
                        factors.append("Transaksi dari luar negeri")
                    if txn['hour'] < 6 or txn['hour'] > 22:
                        factors.append("Transaksi di luar jam normal")
                    if MERCHANT_PROFILES[txn['merchant']]['fraud_risk'] == 'high':
                        factors.append("Merchant berisiko tinggi")
                    if txn['amount_display'] > 5000000:
                        factors.append("Nominal transaksi besar")

                    if factors:
                        for f in factors:
                            st.warning(f)
                    else:
                        st.info("Tidak ada faktor risiko terdeteksi")

                    # Simpan riwayat
                    if 'history' not in st.session_state:
                        st.session_state['history'] = []

                    st.session_state['history'].append({
                        "ID": f"TXN-{random.randint(100000, 999999)}",
                        "Merchant": txn['merchant'],
                        "Nominal": f"Rp {txn['amount_display']:,}",
                        "Lokasi": txn['location'],
                        "Jam": f"{txn['hour']:02d}:00",
                        "Status": "Diblokir" if result['is_fraud'] else "Disetujui",
                        "Kecurigaan": f"{confidence:.1f}%"
                    })

                except Exception as e:
                    st.error("API tidak bisa dihubungi. Pastikan FastAPI jalan!")
        else:
            st.info("Isi form di sebelah kiri dan klik 'Proses Transaksi' untuk melihat hasil analisis.")

with tab2:
    st.markdown("### Performa Model XGBoost")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "XGBoost")
    m2.metric("ROC-AUC", "0.9745")
    m3.metric("Fraud Recall", "86%")
    m4.metric("Fraud Precision", "68%")

    st.markdown("---")
    st.markdown("#### Confusion Matrix")

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
    Cara baca:
    - 56800 transaksi normal diprediksi benar sebagai Normal
    - 84 transaksi fraud berhasil terdeteksi
    - 64 transaksi normal salah diprediksi sebagai Fraud
    - 14 transaksi fraud lolos tidak terdeteksi
    """)

with tab3:
    st.markdown("### Riwayat Transaksi")

    if 'history' in st.session_state and len(st.session_state['history']) > 0:
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df, use_container_width=True)

        total = len(history_df)
        diblokir = history_df[history_df['Status'] == 'Diblokir'].shape[0]
        st.metric("Total Transaksi Sesi Ini", total)
        st.metric("Transaksi Diblokir", diblokir)
    else:
        st.info("Belum ada transaksi. Coba simulasikan transaksi di tab pertama!")