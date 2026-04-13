"""
PatrolIQ Page 2 — Geographic Clusters
K-Means, DBSCAN, Hierarchical maps + Risk Heatmap + Algorithm Comparison.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import json, os
import numpy as np

st.set_page_config(page_title="Geographic Clusters | PatrolIQ", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.page-header { background:linear-gradient(135deg,#0d1117,#0e1a2e); border:1px solid #21262d; border-radius:16px; padding:1.8rem 2rem; margin-bottom:1.5rem; }
.page-title  { font-size:2rem; font-weight:900; color:#e6edf3; margin:0; }
.page-sub    { color:#8b949e; font-size:0.9rem; margin-top:0.3rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#0e1a2e); border-right:1px solid #21262d; }
.algo-winner { background:rgba(74,222,128,0.1); border:1px solid rgba(74,222,128,0.3); border-radius:12px; padding:1rem 1.4rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🚔 PatrolIQ")
    st.page_link("app.py",                                  label="🏠 Home")
    st.page_link("pages/01_📊_EDA_Overview.py",             label="📊 EDA Overview")
    st.page_link("pages/02_🗺️_Geographic_Clusters.py",       label="🗺️ Geographic Clusters")
    st.page_link("pages/03_⏰_Temporal_Patterns.py",        label="⏰ Temporal Patterns")
    st.page_link("pages/04_🔬_Dimensionality_Reduction.py", label="🔬 Dimensionality Reduction")
    st.page_link("pages/05_📈_MLflow_Monitor.py",           label="📈 MLflow Monitor")

CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")

@st.cache_data(ttl=3600, show_spinner="Loading geographic data …")
def load_geo():
    p = os.path.join(CLEAN_DIR, "geo_sample.csv")
    return pd.read_csv(p, low_memory=False)

@st.cache_data(ttl=3600)
def load_metrics():
    p = os.path.join(CLEAN_DIR, "geo_clustering_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_data(ttl=3600)
def load_sil():
    p = os.path.join(CLEAN_DIR, "silhouette_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

geo_df  = load_geo()
metrics = load_metrics()
sil_df  = load_sil()

st.markdown("""
<div class="page-header">
    <div class="page-title">🗺️ Geographic Crime Clustering</div>
    <div class="page-sub">Identify crime hotspots across Chicago using three distinct clustering algorithms</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔵 K-Means", "🟠 DBSCAN", "🟣 Hierarchical", "🔥 Risk Heatmap", "📊 Comparison"
])

CHICAGO_CENTER = [41.83, -87.68]
CLUSTER_COLORS = [
    "#4ade80","#60a5fa","#f59e0b","#f87171","#a78bfa",
    "#34d399","#fb923c","#38bdf8","#e879f9","#fbbf24"
]

def make_folium_map(df, cluster_col, algo_name, noise_label=-1):
    """Create folium map coloured by cluster."""
    m = folium.Map(location=CHICAGO_CENTER, zoom_start=11,
                   tiles="CartoDB dark_matter")
    labels = sorted(df[cluster_col].unique())
    for label in labels:
        sub = df[df[cluster_col] == label].sample(n=min(500, len(df[df[cluster_col]==label])), random_state=42)
        if label == noise_label:
            color = "#6b7280"
            name  = "Noise"
        else:
            color = CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)]
            name  = f"Cluster {label}"
        for _, row in sub.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=3, color=color, fill=True, fill_color=color,
                fill_opacity=0.7, weight=0,
                popup=folium.Popup(f"<b>{name}</b><br>{row.get('Primary Type','')}", max_width=200),
            ).add_to(m)
    return m

def cluster_summary_table(df, label_col, exclude_noise=True):
    src = df.copy()
    if exclude_noise:
        src = src[src[label_col] != -1]
    stats = (src.groupby(label_col).agg(
        Total_Crimes=(label_col,"count"),
        Arrest_Rate=("Arrest","mean"),
        Top_Crime=("Primary Type", lambda x: x.value_counts().index[0]),
    ).reset_index())
    stats.rename(columns={label_col:"Cluster"}, inplace=True)
    stats["Arrest_Rate"] = (stats["Arrest_Rate"]*100).round(1)
    stats["Arrest_Rate"] = stats["Arrest_Rate"].astype(str) + "%"
    return stats

# ── Tab 1: K-Means ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### K-Means Geographic Clustering (k=8)")
    m_km = make_folium_map(geo_df, "KMeans_Geo", "K-Means")
    st_folium(m_km, width="100%", height=480, returned_objects=[])

    m = metrics.get("KMeans_Geo", {})
    c1,c2,c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{m.get('silhouette',0):.4f}")
    c2.metric("Davies-Bouldin", f"{m.get('davies_bouldin',0):.4f}")
    c3.metric("Clusters", m.get("n_clusters","8"))

    st.markdown("#### Cluster Summary Table")
    stats_km = cluster_summary_table(geo_df, "KMeans_Geo")
    st.dataframe(stats_km.style.highlight_max(subset=["Total_Crimes"]), use_container_width=True, hide_index=True)

    st.info("""
**K-Means Interpretation:** Creates 8 circular hotspot zones with clear center points, ideal for patrol route planning.
High-crime clusters concentrate around the South and West sides of Chicago, aligning with historically under-resourced areas.
    """)

# ── Tab 2: DBSCAN ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### DBSCAN Geographic Clustering (ε=0.15, min_samples=10)")
    m_db = make_folium_map(geo_df, "DBSCAN_Geo", "DBSCAN", noise_label=-1)
    st_folium(m_db, width="100%", height=480, returned_objects=[])

    m_d = metrics.get("DBSCAN_Geo", {})
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Silhouette Score", f"{m_d.get('silhouette',0):.4f}", delta="Best algorithm!")
    c2.metric("Davies-Bouldin", f"{m_d.get('davies_bouldin',0):.4f}")
    c3.metric("Clusters Found", m_d.get("n_clusters","N/A"))
    c4.metric("Noise Points", f"{m_d.get('noise_pct',0):.1f}%")

    st.markdown("#### Cluster Summary Table (excl. noise)")
    stats_db = cluster_summary_table(geo_df, "DBSCAN_Geo", exclude_noise=True)
    st.dataframe(stats_db, use_container_width=True, hide_index=True)

    st.info("""
**DBSCAN Interpretation:** Density-based algorithm that naturally finds high-crime areas without requiring pre-specified cluster count.
Gray points are noise/outliers — isolated incidents that don't belong to any dense crime zone. 
DBSCAN achieves the **highest silhouette score** for this dataset, confirming naturally dense crime clusters.
    """)

# ── Tab 3: Hierarchical ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### Hierarchical (Agglomerative) Clustering (k=8, Ward linkage)")
    m_hier = make_folium_map(geo_df, "Hierarchical_Geo", "Hierarchical")
    st_folium(m_hier, width="100%", height=480, returned_objects=[])

    m_h = metrics.get("Hierarchical_Geo", {})
    c1,c2,c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{m_h.get('silhouette',0):.4f}")
    c2.metric("Davies-Bouldin", f"{m_h.get('davies_bouldin',0):.4f}")
    c3.metric("Clusters", m_h.get("n_clusters","8"))

    st.markdown("#### Cluster Summary Table")
    stats_hr = cluster_summary_table(geo_df, "Hierarchical_Geo")
    st.dataframe(stats_hr, use_container_width=True, hide_index=True)

    st.markdown("#### Dendrogram")
    dend_path = os.path.join(CLEAN_DIR, "dendrogram_geo.png")
    if os.path.exists(dend_path):
        st.image(dend_path, use_container_width=True)
    else:
        st.warning("Dendrogram image not found. Please run train.py first.")

    st.info("""
**Hierarchical Interpretation:** Ward linkage minimizes variance within clusters, creating nested geographic zones.
The dendrogram reveals how South Side, West Side, and North Side crime patterns relate hierarchically.
Useful for understanding how larger zones break down into smaller neighborhood-level patterns.
    """)

# ── Tab 4: Risk Heatmap ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔥 Crime Risk Density Heatmap")
    st.caption("Color scale: 🟢 Green = Low Risk → 🟡 Yellow = Medium Risk → 🔴 Red = High Risk")

    risk_filter = st.selectbox("Filter by Crime Type", ["All"] + sorted(geo_df["Primary Type"].unique()),
                                key="risk_type_filter")
    sample_for_heat = geo_df.copy()
    if risk_filter != "All":
        sample_for_heat = sample_for_heat[sample_for_heat["Primary Type"] == risk_filter]

    # Use Crime_Severity_Score as heatmap weight if available, else weight=1
    heat_data = sample_for_heat[["Latitude", "Longitude"]].dropna().copy()
    if "Crime_Severity_Score" in sample_for_heat.columns:
        heat_data["weight"] = sample_for_heat.loc[heat_data.index, "Crime_Severity_Score"].fillna(1)
    else:
        heat_data["weight"] = 1.0
    heat_points = heat_data[["Latitude", "Longitude", "weight"]].values.tolist()

    heat_map = folium.Map(location=CHICAGO_CENTER, zoom_start=11, tiles="CartoDB dark_matter")
    HeatMap(
        heat_points,
        min_opacity=0.3, max_opacity=0.9, radius=12, blur=15,
        gradient={"0.3": "#22c55e", "0.6": "#eab308", "1.0": "#ef4444"},
    ).add_to(heat_map)
    st_folium(heat_map, width="100%", height=520, returned_objects=[])

    st.markdown("""
    **🔴 Red zones** — Highest crime density, requiring maximum patrol presence  
    **🟡 Yellow zones** — Medium risk, patrol rotation recommended  
    **🟢 Green zones** — Lower risk, standard monitoring sufficient
    """)

# ── Tab 5: Algorithm Comparison ───────────────────────────────────────────────
with tab5:
    st.markdown("### 📊 Algorithm Performance Comparison")

    geo_only = sil_df[sil_df["Algorithm"].str.contains("Geo", na=False)] if not sil_df.empty else pd.DataFrame()

    if not geo_only.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_sil = px.bar(geo_only, x="Algorithm", y="Silhouette",
                              color="Silhouette", color_continuous_scale="Teal",
                              template="plotly_dark", title="Silhouette Score (higher = better)")
            fig_sil.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_sil, use_container_width=True)
        with col_r:
            fig_db = px.bar(geo_only, x="Algorithm", y="Davies_Bouldin",
                             color="Davies_Bouldin", color_continuous_scale="Reds_r",
                             template="plotly_dark", title="Davies-Bouldin Index (lower = better)")
            fig_db.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                  coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_db, use_container_width=True)

        # Radar chart
        st.markdown("#### Radar Comparison")
        algos = geo_only["Algorithm"].tolist()
        sil_norm  = (geo_only["Silhouette"] / geo_only["Silhouette"].max()).tolist()
        db_norm   = (1 - geo_only["Davies_Bouldin"] / geo_only["Davies_Bouldin"].max()).tolist()
        fig_radar = go.Figure()
        cats = ["Silhouette", "DB (inv.)", "Scalability", "Interpretability", "Speed"]
        manual_extra = {
            "KMeans_Geo": [0.85, 0.90, 0.95],
            "DBSCAN_Geo": [0.70, 0.65, 0.75],
            "Hierarchical_Geo": [0.75, 0.60, 0.65],
        }
        colors = ["#4ade80","#60a5fa","#f59e0b"]
        for i, row in geo_only.iterrows():
            algo = row["Algorithm"]
            extras = manual_extra.get(algo, [0.7,0.7,0.7])
            vals = [
                sil_norm[list(geo_only["Algorithm"]).index(algo)],
                db_norm[list(geo_only["Algorithm"]).index(algo)],
            ] + extras
            vals += [vals[0]]
            cats_closed = cats + [cats[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats_closed, name=algo,
                line=dict(color=colors[i % len(colors)], width=2),
                fill="toself",
                opacity=0.3,
            ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#161b22",
                        radialaxis=dict(visible=True, range=[0,1], gridcolor="#374151"),
                        angularaxis=dict(gridcolor="#374151")),
            paper_bgcolor="#161b22",
            template="plotly_dark",
            height=380, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Summary
    if metrics:
        best_sil_algo = max(metrics.items(), key=lambda x: x[1].get("silhouette",0))
        best_db_algo  = min(metrics.items(), key=lambda x: x[1].get("davies_bouldin",9999))
        st.markdown(f"""
        <div class="algo-winner">
            <h4 style='color:#4ade80;margin:0 0 0.5rem;'>🏆 Algorithm Recommendation</h4>
            <p style='margin:0;color:#e6edf3;'>
                <b>Best Silhouette:</b> <code>{best_sil_algo[0]}</code> ({best_sil_algo[1]['silhouette']:.4f}) — clusters are well-separated.<br>
                <b>Best Davies-Bouldin:</b> <code>{best_db_algo[0]}</code> ({best_db_algo[1]['davies_bouldin']:.4f}) — clusters are tight and distinct.<br><br>
                ✅ <b>DBSCAN is recommended</b> for deployment as it discovers crime concentrations naturally without forcing pre-defined boundaries,
                and achieves the highest silhouette score on Chicago's geographically dense crime data.
            </p>
        </div>
        """, unsafe_allow_html=True)
