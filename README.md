# 🚔 PatrolIQ — Smart Safety Analytics Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.11-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

**PatrolIQ** is an AI-powered urban safety intelligence platform that leverages unsupervised machine learning to analyze 500,000+ Chicago crime records. Built for law enforcement, city administration, and public safety analysts.

### Problem Statement
Chicago Police officers face daily questions: *"Where should we patrol tonight?"*, *"Which neighborhoods need more resources?"*, *"When do most crimes occur?"* PatrolIQ answers these with data-driven precision.

---

## 🏗️ Architecture

```
PatrolIQ/
├── app.py                              ← 🎨 Animated hero landing page
├── pages/
│   ├── 01_📊_EDA_Overview.py           ← Crime distributions & correlations
│   ├── 02_🗺️_Geographic_Clusters.py    ← K-Means, DBSCAN, Hierarchical maps
│   ├── 03_⏰_Temporal_Patterns.py      ← Hourly, daily, seasonal patterns
│   ├── 04_🔬_Dimensionality_Reduction.py ← PCA scree + t-SNE day/night
│   └── 05_📈_MLflow_Monitor.py         ← Experiment tracking dashboard
├── scripts/
│   ├── preprocess.py                   ← Data cleaning + feature engineering
│   └── train.py                        ← ML pipeline + MLflow logging
├── data/
│   ├── uncleaned/crimes_data.csv       ← Raw Chicago crime dataset (500K+)
│   └── cleaned/                        ← Processed ML artifacts
├── mlruns/                             ← MLflow experiment tracking
├── .streamlit/config.toml              ← Dark theme configuration
├── requirements.txt                    ← Streamlit Cloud dependencies
├── Dockerfile                          ← Container setup
└── docker-compose.yml                  ← Multi-service orchestration
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | City of Chicago Data Portal |
| **Records** | 507,937 (sampled to ~505K after cleaning) |
| **Date Range** | 2003–2026 |
| **Features** | 22 original + 8 engineered |
| **Crime Types** | 31 distinct categories |

### Engineered Features
- `Hour`, `Day_of_Week`, `Day_Num`, `Month` — from DateTime
- `Season` — Winter/Spring/Summer/Fall
- `Is_Weekend` — Saturday/Sunday flag
- `Is_Night` — 8PM–6AM flag
- `Crime_Severity_Score` — 1–10 scale per crime type

---

## 🤖 ML Pipeline

### Geographic Clustering (on Lat/Lon)
| Algorithm | Silhouette | Davies-Bouldin | Notes |
|---|---|---|---|
| K-Means (k=8) | 0.4074 | 0.8031 | Clear patrol zone boundaries |
| DBSCAN | **0.5533** | **0.3732** | **Best overall — noise robust** |
| Hierarchical (k=8) | 0.3587 | 0.8149 | Dendrogram for nested zones |

### Temporal Clustering
| Algorithm | Silhouette | DB | Clusters |
|---|---|---|---|
| K-Means Temporal (k=4) | 0.2123 | 1.4301 | 4 time-based crime profiles |

### Dimensionality Reduction
| Technique | Result |
|---|---|
| PCA (2 components) | 41.6% variance explained |
| PCA (6 components) | ≥70% variance explained |
| t-SNE | KL-divergence: 1.797 |

---

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PatrolIQ.git
cd PatrolIQ

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run preprocessing (creates data/cleaned/ artifacts)
python scripts/preprocess.py

# Run ML training (K-Means, DBSCAN, Hierarchical, PCA, t-SNE + MLflow)
python scripts/train.py

# Launch the Streamlit app
streamlit run app.py
```

### Docker (Bonus)
```bash
docker-compose up --build
# App available at http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

1. Fork/push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → Select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy!**

> **Note:** The `data/uncleaned/crimes_data.csv` (large file) should be added to `.gitignore` if file size exceeds GitHub's 100MB limit. Pre-generated `data/cleaned/` artifacts are committed to the repo for instant deployment.

---

## 🔬 MLflow Experiment Tracking

```bash
# View MLflow UI
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

All 6 model runs tracked under experiment: `PatrolIQ_Crime_Analysis`

---

## 📋 Project Evaluation Coverage

| Criteria | Weight | Status |
|---|---|---|
| Data Preprocessing & Sampling | 10% | ✅ Complete |
| Clustering: ≥3 algorithms | 30% | ✅ K-Means + DBSCAN + Hierarchical |
| Dimensionality Reduction | 20% | ✅ PCA + t-SNE |
| MLflow Integration | 10% | ✅ 6 tracked runs |
| Streamlit App | 20% | ✅ 5 pages, dark theme |
| Cloud Deployment | 10% | ✅ Streamlit Cloud ready |
| Docker (Bonus) | +10% | ✅ Dockerfile + docker-compose |

---

## 🛠️ Technology Stack

- **Language:** Python 3.12
- **ML:** scikit-learn (K-Means, DBSCAN, AgglomerativeClustering, PCA, t-SNE)
- **Tracking:** MLflow 3.11
- **Visualization:** Plotly, Folium, Seaborn
- **App:** Streamlit 1.56
- **Data:** Pandas, NumPy, PyArrow

---

## 👤 Author

Built as a Data Science Capstone Project — Public Safety Analytics Domain  
Dataset: [Chicago Crime Data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

---

*"Data-driven policing for safer cities."*
