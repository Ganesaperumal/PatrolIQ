"""
PatrolIQ - Clustering Analysis Page
Shows K-Means, DBSCAN, Hierarchical results with Silhouette + Davies-Bouldin comparison.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="Clustering Analysis — PatrolIQ", layout="wide", page_icon="🎯")

BASE_DIR = Path(__file__).resolve().parents[2]

# ── Common CSS ────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
html,body,.stApp{background:#0a0e1a;font-family:'Inter',sans-serif;}
.page-title{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#667eea,#f093fb);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem;}
.sub-text{color:#8892b0;font-size:1rem;margin-bottom:2rem;}
.metric-row{display:flex;gap:1rem;margin:1rem 0;}
.mini-card{background:#161b2e;border:1px solid #2a3050;border-radius:12px;padding:1.2rem 1.5rem;flex:1;text-align:center;}
.mini-val{font-size:1.8rem;font-weight:800;color:#667eea;}
.mini-lbl{color:#8892b0;font-size:.8rem;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#161b2e);border-right:1px solid #2a3050;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🎯 Clustering Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Compare K-Means, DBSCAN, and Agglomerative Hierarchical clustering with standardized metrics</div>', unsafe_allow_html=True)

# Load metrics
metrics_path = BASE_DIR / "reports" / "summaries" / "geo_clustering_metrics.json"
temporal_path = BASE_DIR / "reports" / "summaries" / "temporal_clustering_metrics.json"
data_path = BASE_DIR / "data" / "processed" / "model_ready_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path, low_memory=False)

try:
    df = load_data()
except:
    df = None

tab1, tab2 = st.tabs(["📍 Geographic Clustering", "⏰ Temporal Clustering"])

with tab1:
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Algorithm comparison chart
        st.subheader("Algorithm Performance Comparison")
        alg_names = list(metrics.keys())
        sil_scores = [metrics[k].get("silhouette", 0) or 0 for k in alg_names]
        db_scores  = [metrics[k].get("davies_bouldin", 0) or 0 for k in alg_names]

        col1, col2 = st.columns(2)
        with col1:
            fig_sil = px.bar(x=[a.upper() for a in alg_names], y=sil_scores,
                             labels={"x": "Algorithm", "y": "Silhouette Score"},
                             title="Silhouette Score (Higher = Better)",
                             color=sil_scores, color_continuous_scale="Viridis")
            fig_sil.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e",
                                  font_color="#ccd6f6", showlegend=False)
            fig_sil.add_hline(y=0.5, line_dash="dash", line_color="#f093fb",
                              annotation_text="Target: 0.5")
            st.plotly_chart(fig_sil, use_container_width=True)

        with col2:
            fig_db = px.bar(x=[a.upper() for a in alg_names], y=db_scores,
                            labels={"x": "Algorithm", "y": "Davies-Bouldin Index"},
                            title="Davies-Bouldin Index (Lower = Better)",
                            color=db_scores, color_continuous_scale="RdYlGn_r")
            fig_db.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6", showlegend=False)
            st.plotly_chart(fig_db, use_container_width=True)

        # Metrics summary table
        st.subheader("📊 Detailed Metrics Summary")
        rows = []
        for alg, vals in metrics.items():
            rows.append({
                "Algorithm": alg.upper(),
                "Silhouette Score": f"{vals.get('silhouette', 'N/A'):.4f}" if vals.get('silhouette') is not None else "N/A",
                "Davies-Bouldin": f"{vals.get('davies_bouldin', 'N/A'):.4f}" if vals.get('davies_bouldin') is not None else "N/A",
                "Clusters Found": vals.get('n_clusters', 'N/A'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Dendrogram
        dendro_path = BASE_DIR / "reports" / "figures" / "dendrogram_geo.png"
        if dendro_path.exists():
            st.subheader("🌳 Hierarchical Clustering Dendrogram")
            st.image(str(dendro_path), use_container_width=True)
    else:
        st.warning("⚠️ Run `src/models/geo_clustering.py` first to see clustering results.")

    # Scatter plot of clusters (sample data from model_ready)
    if df is not None and "Latitude" in df.columns:
        st.subheader("🗺️ Geographic Cluster Scatter")
        import joblib
        model_path = BASE_DIR / "models" / "geo_cluster.pkl"
        if model_path.exists():
            sample = df.dropna(subset=["Latitude","Longitude"]).sample(min(20_000, len(df)), random_state=42)
            km = joblib.load(model_path)
            sample["KMeans_Cluster"] = km.predict(sample[["Latitude","Longitude"]])
            fig_scatter = px.scatter_mapbox(sample, lat="Latitude", lon="Longitude",
                color="KMeans_Cluster", hover_name="Primary Type" if "Primary Type" in sample.columns else None,
                zoom=9, height=500, mapbox_style="carto-darkmatter",
                title="K-Means Geographic Clusters")
            fig_scatter.update_layout(paper_bgcolor="#0a0e1a", font_color="#ccd6f6")
            st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    if temporal_path.exists():
        with open(temporal_path) as f:
            t_metrics = json.load(f)

        # Temporal metrics
        t_names  = list(t_metrics.keys())
        t_sil    = [t_metrics[k].get("silhouette", 0) for k in t_names]
        t_db     = [t_metrics[k].get("davies_bouldin", 0) for k in t_names]

        col1, col2 = st.columns(2)
        with col1:
            fig_ts = px.bar(x=t_names, y=t_sil, title="Temporal Silhouette Scores",
                            color=t_sil, color_continuous_scale="Plasma")
            fig_ts.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6", showlegend=False)
            st.plotly_chart(fig_ts, use_container_width=True)

        with col2:
            fig_td = px.bar(x=t_names, y=t_db, title="Temporal Davies-Bouldin Scores",
                            color=t_db, color_continuous_scale="RdYlGn_r")
            fig_td.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6", showlegend=False)
            st.plotly_chart(fig_td, use_container_width=True)

        # Elbow plot
        elbow_path = BASE_DIR / "reports" / "summaries" / "temporal_elbow_data.csv"
        if elbow_path.exists():
            elbow_df = pd.read_csv(elbow_path)
            st.subheader("📈 Elbow Method — Optimal k Selection")
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=elbow_df["k"], y=elbow_df["silhouette"],
                mode="lines+markers", name="Silhouette Score", line=dict(color="#667eea", width=3)))
            fig_elbow.update_layout(title="Silhouette Score vs k",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e",
                font_color="#ccd6f6", xaxis_title="k (clusters)", yaxis_title="Silhouette Score")
            st.plotly_chart(fig_elbow, use_container_width=True)
    else:
        st.warning("⚠️ Run `src/models/temporal_clustering.py` first.")
