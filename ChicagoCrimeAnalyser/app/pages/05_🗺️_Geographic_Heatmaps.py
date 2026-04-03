"""
PatrolIQ - Geographic Heatmaps Page
Algorithm selector with Plotly scatter mapbox + inline Folium heatmap
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(page_title="Geographic Heatmaps — PatrolIQ", layout="wide", page_icon="🗺️")
BASE_DIR = Path(__file__).resolve().parents[2]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
html,body,.stApp{background:#0a0e1a;}
.page-title{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#fa709a,#fee140);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem;}
.sub-text{color:#8892b0;margin-bottom:2rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#161b2e);border-right:1px solid #2a3050;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🗺️ Geographic Heatmaps</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Visualize crime hotspots across Chicago using different clustering algorithms</div>', unsafe_allow_html=True)

DATA_PATH = BASE_DIR / "data" / "processed" / "model_ready_data.csv"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"

@st.cache_data
def load_geo_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df.dropna(subset=["Latitude","Longitude"])

# Sidebar controls
with st.sidebar:
    st.markdown("### 🗺️ Map Controls")
    algorithm = st.selectbox("Clustering Algorithm",
                             ["K-Means (k=9)", "DBSCAN (density-based)", "Crime Severity"])
    map_sample = st.slider("Sample size", 5_000, 50_000, 20_000, step=5000)
    map_style  = st.selectbox("Map Style",
                              ["carto-darkmatter","open-street-map","carto-positron","stamen-toner"])
    if "Primary Type" in pd.read_csv(DATA_PATH, nrows=1).columns:
        crime_filter = st.text_input("Filter by crime type (leave blank for all)")

try:
    df = load_geo_data()
    sample = df.sample(min(map_sample, len(df)), random_state=42)
    if 'crime_filter' in locals() and crime_filter:
        sample = sample[sample["Primary Type"].str.contains(crime_filter, case=False, na=False)]

    # Choose color column
    if algorithm.startswith("K-Means"):
        km_path = MODELS_DIR / "geo_cluster.pkl"
        if km_path.exists():
            km = joblib.load(km_path)
            sample = sample.copy()
            sample["Cluster"] = km.predict(sample[["Latitude","Longitude"]])
            color_col = "Cluster"
            title = "K-Means Geographic Hotspot Clusters (k=9)"
        else:
            color_col = "Crime_Severity_Score" if "Crime_Severity_Score" in sample.columns else None
            title = "Crime Severity Map (K-Means model not found — run geo_clustering.py)"
    elif algorithm.startswith("DBSCAN"):
        dbscan_html = FIG_DIR / "map_dbscan_geo.html"
        if dbscan_html.exists():
            st.subheader("DBSCAN Cluster Map (Folium)")
            with open(dbscan_html, "r") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
            color_col = "Crime_Severity_Score" if "Crime_Severity_Score" in sample.columns else None
            title = "DBSCAN — Crime Density Map"
        else:
            st.warning("Run `geo_clustering.py` to generate the DBSCAN Folium map.")
            color_col = None; title = ""
    else:
        color_col = "Crime_Severity_Score" if "Crime_Severity_Score" in sample.columns else None
        title = "Crime Severity Geographic Distribution"

    # Main plotly mapbox
    if color_col is not None:
        fig = px.scatter_mapbox(
            sample, lat="Latitude", lon="Longitude",
            color=color_col,
            hover_name="Primary Type" if "Primary Type" in sample.columns else None,
            hover_data={"Latitude": False, "Longitude": False,
                        "Hour": True if "Hour" in sample.columns else False},
            zoom=9, height=600, mapbox_style=map_style,
            color_continuous_scale="Inferno",
            title=title
        )
        fig.update_layout(paper_bgcolor="#0a0e1a", font_color="#ccd6f6",
                          margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    # Cluster center markers (K-Means only)
    centers_path = BASE_DIR / "reports" / "summaries" / "kmeans_geo_centers.csv"
    if algorithm.startswith("K-Means") and centers_path.exists():
        centers = pd.read_csv(centers_path)
        centers["Label"] = [f"Cluster {i}" for i in range(len(centers))]
        st.subheader("📌 K-Means Cluster Centers")
        fig_c = px.scatter_mapbox(centers, lat="Latitude", lon="Longitude",
                                  text="Label", zoom=9, height=350, mapbox_style=map_style,
                                  size=[1]*len(centers))
        fig_c.update_traces(marker_size=15, marker_color="#f093fb")
        fig_c.update_layout(paper_bgcolor="#0a0e1a", font_color="#ccd6f6",
                            margin={"r":0,"t":20,"l":0,"b":0})
        st.plotly_chart(fig_c, use_container_width=True)

    # Final EDA Folium heatmap
    heatmap_html = FIG_DIR / "crime_heatmap.html"
    if heatmap_html.exists():
        with st.expander("🔥 Raw Crime Density Heatmap (Folium — EDA)"):
            import streamlit.components.v1 as components
            with open(heatmap_html, "r") as f:
                components.html(f.read(), height=500, scrolling=True)

except FileNotFoundError:
    st.error("Data not found. Please run the preprocessing and feature engineering scripts first.")
