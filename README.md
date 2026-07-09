# Real-Time Fraud Detection System

Sistem deteksi transaksi keuangan mencurigakan secara real-time menggunakan Machine Learning, dibangun dengan arsitektur end-to-end production-ready.

## Latar Belakang

Industri keuangan global kehilangan miliaran dolar setiap tahun akibat fraud. Proyek ini mensimulasikan sistem yang digunakan perusahaan fintech dan perbankan untuk mendeteksi transaksi mencurigakan secara otomatis sebelum kerugian terjadi.

## Arsitektur Sistem

```
Dataset (Kaggle) -> EDA & Preprocessing -> Model Training (MLflow) -> FastAPI -> Streamlit Dashboard
```

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Modeling | Python, Pandas, Scikit-learn, XGBoost |
| Imbalanced Handling | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Version Control | Git + GitHub |

## Hasil Model

| Model | ROC-AUC | Fraud Recall | Fraud Precision |
|---|---|---|---|
| Logistic Regression | 0.9698 | 92% | 6% |
| Random Forest | 0.9688 | 82% | 82% |
| XGBoost | 0.9745 | 86% | 68% |

XGBoost dipilih karena memiliki keseimbangan terbaik antara precision dan recall.

## Cara Menjalankan

### 1. Clone repo
```bash
git clone https://github.com/richardvanuella/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Setup environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download dataset
Download `creditcard.csv` dari [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dan taruh di `data/raw/`

### 4. Train model
Jalankan notebook `notebooks/02_modeling.ipynb` untuk melatih model dan menyimpannya ke `models/`

### 5. Jalankan FastAPI
```bash
uvicorn src.main:app --reload
```

### 6. Jalankan Dashboard
```bash
python -m streamlit run src/dashboard.py
```

Buka browser di `http://localhost:8501`

## Struktur Project

```
fraud-detection-system/
├── data/
│   └── raw/              # Dataset (tidak di-push ke GitHub)
├── models/               # Model hasil training (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   └── 02_modeling.ipynb # Preprocessing & Model Training
├── src/
│   ├── main.py           # FastAPI backend
│   └── dashboard.py      # Streamlit dashboard
├── .gitignore
├── requirements.txt
└── README.md
```

## Key Findings

- Dataset sangat imbalanced: hanya 0.17% transaksi adalah fraud
- Ditangani dengan SMOTE untuk oversample kelas minoritas
- Fitur paling berpengaruh: V17, V14, V12, V10 (korelasi negatif kuat dengan fraud)
- Model mampu mendeteksi 86% dari seluruh transaksi fraud yang ada