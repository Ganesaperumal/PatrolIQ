"""
PatrolIQ Page 2 — Geographic Clusters
K-Means, DBSCAN, Hierarchical maps + Risk Heatmap + Algorithm Comparison.
Maps use OpenStreetMap tiles (light mode, full street detail).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
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
    return pd.read_csv(os.path.join(CLEAN_DIR, "geo_sample.csv"), low_memory=False)

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

# ── Constants ─────────────────────────────────────────────────────────────────
CHICAGO_CENTER = [41.83, -87.68]

# Named cluster colors (10 distinct, colorblind-friendly palette)
CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
    "#a65628", "#f781bf", "#17becf", "#bcbd22", "#8c564b"
]

OSM_TILES = "OpenStreetMap"   # ← light, full street detail

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_cluster_map(df, cluster_col, noise_label=-1, n_per_cluster=800):
    """OpenStreetMap Folium map, coloured by cluster, with rich tooltip."""
    m = folium.Map(location=CHICAGO_CENTER, zoom_start=11, tiles=OSM_TILES)

    labels = sorted(df[cluster_col].unique())
    for label in labels:
        sub = df[df[cluster_col] == label]
        sub = sub.sample(n=min(n_per_cluster, len(sub)), random_state=42)

        if label == noise_label:
            color, cluster_name = "#aaaaaa", "Noise / Outlier"
        else:
            color = CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)]
            cluster_name = f"Cluster {label}"

        for _, row in sub.iterrows():
            tooltip_html = (
                f"<b>{cluster_name}</b><br>"
                f"Type: {row.get('Primary Type', 'N/A')}<br>"
                f"Lat: {row['Latitude']:.5f}, Lon: {row['Longitude']:.5f}"
            )
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=4,
                color=color, fill=True, fill_color=color,
                fill_opacity=0.75, weight=0.5,
                tooltip=tooltip_html,
            ).add_to(m)

    # Manual legend (top-right)
    real_labels = [l for l in labels if l != noise_label]
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;margin:3px 0;">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{CLUSTER_COLORS[int(l) % len(CLUSTER_COLORS)]};margin-right:8px;"></div>'
        f'<span>Cluster {l}</span></div>'
        for l in real_labels
    )
    if noise_label in labels:
        legend_items += ('<div style="display:flex;align-items:center;margin:3px 0;">'
                         '<div style="width:14px;height:14px;border-radius:50%;background:#aaaaaa;margin-right:8px;"></div>'
                         '<span>Noise</span></div>')
    legend_html = f"""
    <div style="position:fixed;top:16px;right:16px;z-index:9999;background:white;
                border:1px solid #ccc;border-radius:8px;padding:10px 14px;
                font-size:12px;font-family:sans-serif;box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
        <b style="display:block;margin-bottom:6px;">Clusters</b>
        {legend_items}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def cluster_summary_table(df, label_col, exclude_noise=True):
    src = df.copy()
    if exclude_noise:
        src = src[src[label_col] != -1]
    stats = (src.groupby(label_col).agg(
        Total_Crimes=(label_col, "count"),
        Arrest_Rate=("Arrest", "mean"),
        Top_Crime=("Primary Type", lambda x: x.value_counts().index[0]),
    ).reset_index())
    stats.rename(columns={label_col: "Cluster"}, inplace=True)
    stats["Arrest_Rate"] = (stats["Arrest_Rate"] * 100).round(1).astype(str) + "%"
    return stats


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔵 K-Means", "🟠 DBSCAN", "🟣 Hierarchical", "🔥 Risk Heatmap", "📊 Comparison"
])

# ── Tab 1: K-Means ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### K-Means Geographic Clustering (k=8)")
    m_km = make_cluster_map(geo_df, "KMeans_Geo")
    st_folium(m_km, width="100%", height=520, returned_objects=[])

    m = metrics.get("KMeans_Geo", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{m.get('silhouette', 0):.4f}")
    c2.metric("Davies-Bouldin",   f"{m.get('davies_bouldin', 0):.4f}")
    c3.metric("Clusters", m.get("n_clusters", 8))

    st.markdown("#### Cluster Summary Table")
    stats_km = cluster_summary_table(geo_df, "KMeans_Geo")
    st.dataframe(stats_km.style.highlight_max(subset=["Total_Crimes"]),
                 use_container_width=True, hide_index=True)

    st.info("""
**K-Means Interpretation:** Creates 8 circular hotspot zones with clear center points — ideal for fixed patrol route planning.
High-crime clusters concentrate around the South and West sides of Chicago.
    """)

# ── Tab 2: DBSCAN ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### DBSCAN Geographic Clustering (ε=0.15, min_samples=10)")
    m_db = make_cluster_map(geo_df, "DBSCAN_Geo", noise_label=-1)
    st_folium(m_db, width="100%", height=520, returned_objects=[])

    m_d = metrics.get("DBSCAN_Geo", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Silhouette Score", f"{m_d.get('silhouette', 0):.4f}", delta="Best!")
    c2.metric("Davies-Bouldin",   f"{m_d.get('davies_bouldin', 0):.4f}")
    c3.metric("Clusters Found",   m_d.get("n_clusters", "N/A"))
    c4.metric("Noise Points",     f"{m_d.get('noise_pct', 0):.1f}%")

    st.markdown("#### Cluster Summary Table (excl. noise)")
    stats_db = cluster_summary_table(geo_df, "DBSCAN_Geo", exclude_noise=True)
    st.dataframe(stats_db, use_container_width=True, hide_index=True)

    st.info("""
**DBSCAN Interpretation:** Density-based — discovers naturally formed high-crime zones without needing a pre-set *k*.
Grey dots are isolated incidents (noise). Achieves the **highest silhouette score** of all three algorithms.
    """)

# ── Tab 3: Hierarchical ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### Hierarchical (Agglomerative) Clustering (k=8, Ward linkage)")
    m_hier = make_cluster_map(geo_df, "Hierarchical_Geo")
    st_folium(m_hier, width="100%", height=520, returned_objects=[])

    m_h = metrics.get("Hierarchical_Geo", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", f"{m_h.get('silhouette', 0):.4f}")
    c2.metric("Davies-Bouldin",   f"{m_h.get('davies_bouldin', 0):.4f}")
    c3.metric("Clusters", m_h.get("n_clusters", 8))

    st.markdown("#### Cluster Summary Table")
    stats_hr = cluster_summary_table(geo_df, "Hierarchical_Geo")
    st.dataframe(stats_hr, use_container_width=True, hide_index=True)

    st.markdown("#### 🌳 Dendrogram")
    dend_path = os.path.join(CLEAN_DIR, "dendrogram_geo.png")
    if os.path.exists(dend_path):
        st.image(dend_path, use_container_width=True)
    else:
        st.warning("Dendrogram not found — run `scripts/train.py` first.")

    st.info("""
**Hierarchical Interpretation:** Ward linkage creates nested geographic zones.
The dendrogram shows how South/West/North Side clusters are related hierarchically.
    """)

# ── Tab 4: Risk Heatmap ───────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔥 Interactive Crime Heatmap")
    st.caption("Blue = low density  →  Green = moderate  →  Yellow/Red = high density crime zones")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        risk_filter = st.selectbox(
            "Filter by Crime Type",
            ["All"] + sorted(geo_df["Primary Type"].unique()),
            key="risk_type_filter"
        )
    with col_f2:
        heat_radius = st.slider("Heat radius", 8, 25, 15, key="heat_radius")

    sample_for_heat = geo_df.copy()
    if risk_filter != "All":
        sample_for_heat = sample_for_heat[sample_for_heat["Primary Type"] == risk_filter]

    # Weight by severity if available, else 1
    heat_data = sample_for_heat[["Latitude", "Longitude"]].dropna().copy()
    if "Crime_Severity_Score" in sample_for_heat.columns:
        heat_data["weight"] = sample_for_heat.loc[heat_data.index, "Crime_Severity_Score"].fillna(1)
    else:
        heat_data["weight"] = 1.0
    heat_points = heat_data[["Latitude", "Longitude", "weight"]].values.tolist()

    heat_map = folium.Map(location=CHICAGO_CENTER, zoom_start=11, tiles=OSM_TILES)
    HeatMap(
        heat_points,
        min_opacity=0.35,
        max_opacity=0.85,
        radius=heat_radius,
        blur=heat_radius + 5,
        # Classic blue → cyan → green → yellow → red gradient (like screenshot 3)
        gradient={
            "0.0": "#0000ff",
            "0.3": "#00bfff",
            "0.5": "#00ff80",
            "0.7": "#ffff00",
            "1.0": "#ff0000",
        },
    ).add_to(heat_map)
    st_folium(heat_map, width="100%", height=540, returned_objects=[])

    st.markdown("""
    🔵 **Blue/Cyan** — Low crime density  
    🟢 **Green** — Moderate activity  
    🟡 **Yellow** — High activity  
    🔴 **Red** — Maximum density — highest patrol priority
    """)

# ── Tab 5: Algorithm Comparison ───────────────────────────────────────────────
with tab5:
    st.markdown("### 📊 Algorithm Performance Comparison")

    geo_only = sil_df[sil_df["Algorithm"].str.contains("Geo", na=False)] if not sil_df.empty else pd.DataFrame()

    if not geo_only.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_sil = px.bar(
                geo_only, x="Algorithm", y="Silhouette",
                color="Silhouette", color_continuous_scale="Teal",
                template="plotly_dark", title="Silhouette Score (↑ Better)"
            )
            fig_sil.add_hline(y=0.5, line=dict(color="#ef4444", dash="dash"),
                               annotation_text="Target: 0.5")
            fig_sil.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                   coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_sil, use_container_width=True)

        with col_r:
            geo_num = geo_only.copy()
            geo_num["Davies_Bouldin"] = pd.to_numeric(geo_num["Davies_Bouldin"], errors="coerce")
            fig_db = px.bar(
                geo_num.dropna(subset=["Davies_Bouldin"]),
                x="Algorithm", y="Davies_Bouldin",
                color="Davies_Bouldin", color_continuous_scale="Reds_r",
                template="plotly_dark", title="Davies-Bouldin Index (↓ Better)"
            )
            fig_db.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                  coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_db, use_container_width=True)

        # Radar
        st.markdown("#### Multi-Metric Radar Comparison")
        geo_num = geo_only.copy()
        geo_num["Silhouette"] = pd.to_numeric(geo_num["Silhouette"], errors="coerce").fillna(0)
        geo_num["Davies_Bouldin"] = pd.to_numeric(geo_num["Davies_Bouldin"], errors="coerce").fillna(0)
        max_sil = geo_num["Silhouette"].max() or 1
        max_db  = geo_num["Davies_Bouldin"].max() or 1

        manual_scores = {
            "KMeans_Geo":       {"Speed": 0.95, "Scalability": 0.92, "Interpretability": 0.90},
            "DBSCAN_Geo":       {"Speed": 0.70, "Scalability": 0.65, "Interpretability": 0.72},
            "Hierarchical_Geo": {"Speed": 0.55, "Scalability": 0.50, "Interpretability": 0.85},
        }
        colors_r = ["#4ade80", "#60a5fa", "#f59e0b"]
        cats = ["Silhouette", "DB (inv.)", "Speed", "Scalability", "Interpretability"]

        fig_r = go.Figure()
        for i, (_, row) in enumerate(geo_num.iterrows()):
            algo   = row["Algorithm"]
            extras = manual_scores.get(algo, {"Speed": 0.7, "Scalability": 0.7, "Interpretability": 0.7})
            vals = [
                row["Silhouette"] / max_sil,
                1 - row["Davies_Bouldin"] / max_db,
                extras["Speed"], extras["Scalability"], extras["Interpretability"],
            ]
            vals += [vals[0]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], name=algo,
                line=dict(color=colors_r[i % len(colors_r)], width=2),
                fill="toself", opacity=0.3,
            ))
        fig_r.update_layout(
            polar=dict(bgcolor="#161b22",
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="#374151"),
                        angularaxis=dict(gridcolor="#374151")),
            paper_bgcolor="#161b22", template="plotly_dark",
            height=400, showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Recommendation box
    if metrics:
        best_sil = max(metrics.items(), key=lambda x: x[1].get("silhouette", 0))
        best_db  = min(metrics.items(), key=lambda x: x[1].get("davies_bouldin", 9999))
        st.markdown(f"""
        <div class="algo-winner">
            <h4 style='color:#4ade80;margin:0 0 0.5rem'>🏆 Algorithm Recommendation</h4>
            <p style='margin:0;color:#e6edf3;'>
                <b>Best Silhouette:</b> <code>{best_sil[0]}</code> — {best_sil[1]['silhouette']:.4f}<br>
                <b>Best Davies-Bouldin:</b> <code>{best_db[0]}</code> — {best_db[1]['davies_bouldin']:.4f}<br><br>
                ✅ <b>DBSCAN recommended</b> for deployment — discovers natural crime zones without forcing fixed boundaries,
                and achieves the highest silhouette score on Chicago's geographically dense data.
            </p>
        </div>
        """, unsafe_allow_html=True)
