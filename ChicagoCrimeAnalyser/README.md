# PatrolIQ - Chicago Crime Analytics Platform

The definitive version of the Chicago Police Department urban safety intelligence platform. Built to process 500k records, establish temporal and spatial clustering, and display everything through an interactable Streamlit dashboard.

## 🚀 Key Capabilities

- **Stratified Preprocessing**: Robust geographic bounding, exact temporal cyclical mappings, and dataset compression for highly rapid iteration.
- **Geographic Clustering**: Identifies K-Means Hotspots, density areas via DBSCAN, and hierarchical associations across Chicago. Detailed scoring (Silhouette/Davies-Bouldin Index) and `.html` Folium Mapping are outputted dynamically.
- **Dimensionality Reduction**: Visualized 45+ feature correlations into dense point grids utilizing **PCA (w/ Scree Plots)**, **t-SNE**, and **UMAP**.
- **Temporal Analysis**: Provides heatmap mappings showing exactly when crime peaks based on weekday distributions.
- **MLflow Tracking**: Complete backend tracking node recording clustering properties dynamically.

## 📁 Repository Structure
```
PatrolIQ/
├── data/
│   ├── unprocessed/          # Raw dataset location
│   ├── processed/            # Cleaned ML-ready structures
│   └── figures/              # Exported HTML/PNG EDA plots
├── models/                   # Auto-exported .pkl trained models
├── reports/
│   ├── figures/              # ML-generated static plots & Dendrograms
│   └── summaries/            # Output tables & Json Metrics
├── src/
│   ├── data_preprocessing/   # Cleaning & Quality Validations
│   ├── analysis/             # Feature Engineering & EDA Generation 
│   └── models/               # Cluter Training, Dim-Reduction, and MLflow Trackers
└── app/
    ├── 🏠_Home.py            # Streamlit Dashboard Primary Entrypoint
    └── pages/                # Multi-page analytics (Clustering, PCA, MLflow)
```

## 🛠️ Installation & Setup

1. Setup the Python Virtual Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate the Required Directories and Run the Pipeline:
   (Note: If you have already run these scripts from `src/`, this step is unnecessary)
```bash
mkdir -p data/processed reports/summaries reports/figures models
python src/data_preprocessing/clean_data.py
python src/data_preprocessing/validate_data.py
python src/analysis/eda_pipeline.py
python src/analysis/feature_engineering.py
python src/models/geo_clustering.py
python src/models/temporal_clustering.py
python src/models/dimensionality_reduction.py
```

3. Launch the Streamlit Intelligence Platform:
```bash
streamlit run app/🏠_Home.py
```

## 🐳 Docker Deployment
Run this completely isolated in a containerized environment!

```bash
docker-compose up --build
```
Navigate to `http://localhost:8501` to view your dashboard!

> Created for the Chicago Police Department Resource Allocation Project.
