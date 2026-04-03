# PatrolIQ - Smart Safety Analytics Platform

An intelligent urban safety platform analyzing Chicago crime data to optimize police resource allocation. The platform leverages unsupervised machine learning techniques to find crime hotspots and visualize complex multi-dimensional crime patterns.

---

## Features

- **Geographic Hotspot Clustering:** Utilizes K-Means, DBSCAN, and Agglomerative Clustering to spatially identify density risk zones.
- **Temporal Pattern Analysis:** Captures cyclical temporal structures (hour of day/monthly seasonality) to identify when resources are needed most.
- **Dimensionality Reduction:** Compresses 40+ structured features into visual 3D/2D space using PCA and t-SNE components.
- **Automated Experiment Tracking:** Integrated `MLflow` engine stores optimal hyper-parameters and models for reproducibility.
- **Streamlit Interactive UI:** A high-performance, dynamic dashboard for analysts to explore the resulting data.

## File Structure

```
PatrolIQ/
├── app/                  # Streamlit Dashboard application
│   └── app.py
├── data/                 # Data Directory (Ignored from remote)
│   ├── unprocessed/      # Raw Chicago Dataset
│   ├── processed/        # Cleaned and ML-ready Data
│   └── figures/          # Exploratory Data Analysis PNGs
├── mlruns/               # MLflow local tracking records
├── notebooks/            # EDA python scripts/notebooks
│   └── eda.py
├── src/                  # Source Code Modules
│   ├── data_prep.py      # Cleans Raw Dataset
│   ├── feature_engine.py # Adds ML coordinates, encodings
│   └── models.py         # The Clustering/PCA Pipeline
├── Dockerfile            # Container configuration
└── requirements.txt      # Python Dependencies
```

## Running the Application Locally

1. Create a virtual environment and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit application:
   ```bash
   streamlit run app/app.py
   ```

## Running via Docker

For easy and replicable production deployment:
```bash
docker build -t patroliq-app .
docker run -p 8501:8501 patroliq-app
```
Enjoy the platform!
