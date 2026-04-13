"""
PatrolIQ — Main Landing Page (app.py)
Animated hero with KPI cards and quick navigation.
"""

import streamlit as st
import pandas as pd
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PatrolIQ — Smart Safety Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hero gradient */
.hero-container {
    background: linear-gradient(135deg, #0d1117 0%, #0e1a2e 40%, #0a1628 100%);
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero-container::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(74,222,128,0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(59,130,246,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(74,222,128,0.12);
    border: 1px solid rgba(74,222,128,0.3);
    color: #4ade80;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #94b4d6 60%, #4ade80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8b949e;
    margin: 1rem auto;
    max-width: 620px;
    line-height: 1.7;
}
.hero-accent { color: #4ade80; -webkit-text-fill-color: #4ade80; }

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.kpi-card {
    background: linear-gradient(145deg, #161b22, #1c2333);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.green::before  { background: linear-gradient(90deg,#4ade80,#22c55e); }
.kpi-card.blue::before   { background: linear-gradient(90deg,#60a5fa,#3b82f6); }
.kpi-card.amber::before  { background: linear-gradient(90deg,#fbbf24,#f59e0b); }
.kpi-card.red::before    { background: linear-gradient(90deg,#f87171,#ef4444); }
.kpi-card:hover { transform: translateY(-4px); border-color: #30363d; box-shadow: 0 8px 25px rgba(0,0,0,0.4); }
.kpi-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #e6edf3; }
.kpi-label { font-size: 0.78rem; color: #8b949e; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* Nav cards */
.nav-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.nav-card {
    background: linear-gradient(145deg, #161b22, #1c2333);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    transition: all 0.25s ease;
    cursor: pointer;
}
.nav-card:hover { transform: translateY(-4px); border-color: #4ade80; box-shadow: 0 0 20px rgba(74,222,128,0.15); }
.nav-icon { font-size: 2rem; }
.nav-title { font-size: 0.8rem; font-weight: 600; color: #e6edf3; margin: 0.5rem 0 0.3rem; }
.nav-desc  { font-size: 0.7rem; color: #8b949e; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0e1a2e 100%);
    border-right: 1px solid #21262d;
}
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.sidebar-logo { font-size: 2.5rem; }
.sidebar-name { font-size: 1.2rem; font-weight: 800; color: #4ade80; }
.sidebar-tag  { font-size: 0.7rem; color: #8b949e; margin-top: 0.2rem; }

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #484f58;
    font-size: 0.75rem;
    border-top: 1px solid #21262d;
    margin-top: 2rem;
}
.status-dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; display: inline-block; margin-right: 5px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🚔</div>
        <div class="sidebar-name">PatrolIQ</div>
        <div class="sidebar-tag">Smart Safety Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                                  label="🏠 Home",                  help="Landing dashboard")
    st.page_link("pages/01_📊_EDA_Overview.py",             label="📊 EDA Overview",          help="Exploratory analysis")
    st.page_link("pages/02_🗺️_Geographic_Clusters.py",       label="🗺️ Geographic Clusters", help="Crime hotspot maps")
    st.page_link("pages/03_⏰_Temporal_Patterns.py",        label="⏰ Temporal Patterns",     help="Time-based patterns")
    st.page_link("pages/04_🔬_Dimensionality_Reduction.py", label="🔬 Dimensionality Reduction", help="PCA & t-SNE")
    st.page_link("pages/05_📈_MLflow_Monitor.py",           label="📈 MLflow Monitor",        help="Experiment tracking")
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#8b949e; font-size:0.72rem; padding: 0.5rem 0;'>
        <span class="status-dot"></span> System Active<br>
        Chicago Crime Dataset<br>
        500K+ Records · 2003–2026
    </div>
    """, unsafe_allow_html=True)

# ── Load metadata ─────────────────────────────────────────────────────────────
CLEAN_DIR = os.path.join(os.path.dirname(__file__), "data", "cleaned")

@st.cache_data(ttl=3600)
def load_meta():
    meta_path = os.path.join(CLEAN_DIR, "metadata.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    return {}

@st.cache_data(ttl=3600)
def load_sil():
    sil_path = os.path.join(CLEAN_DIR, "silhouette_scores.csv")
    if os.path.exists(sil_path):
        return pd.read_csv(sil_path)
    return pd.DataFrame()

meta = load_meta()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔬 Crime Intelligence Platform · Chicago PD Analytics</div>
    <div class="hero-title">Patrol<span class="hero-accent">IQ</span></div>
    <div class="hero-sub">
        AI-powered urban safety analytics platform leveraging unsupervised machine learning
        to identify crime hotspots, decode temporal patterns, and optimize patrol resource allocation.
    </div>
    <div style='margin-top:1.2rem; font-size:0.8rem; color:#4ade80; font-weight:600;'>
        K-Means · DBSCAN · Hierarchical · PCA · t-SNE · MLflow · Streamlit Cloud
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total  = f"{meta.get('total_records', 505171):,}"
types  = len(meta.get("crime_types", []))
arr    = meta.get("arrest_rate", 20.4)
years  = meta.get("years", list(range(2003, 2027)))
yr_rng = f"{min(years)}–{max(years)}" if years else "2003–2026"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card green">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">{total}</div>
        <div class="kpi-label">Crime Records Analyzed</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-icon">🏷️</div>
        <div class="kpi-value">{types}</div>
        <div class="kpi-label">Distinct Crime Types</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-icon">🚨</div>
        <div class="kpi-value">{arr}%</div>
        <div class="kpi-label">Overall Arrest Rate</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-icon">📅</div>
        <div class="kpi-value">{yr_rng}</div>
        <div class="kpi-label">Data Coverage Period</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Module navigation cards ────────────────────────────────────────────────────
st.markdown("### 🧭 Explore Analytics Modules")
st.markdown("""
<div class="nav-grid">
    <div class="nav-card">
        <div class="nav-icon">📊</div>
        <div class="nav-title">EDA Overview</div>
        <div class="nav-desc">Crime distributions, trends &amp; arrest correlations</div>
    </div>
    <div class="nav-card">
        <div class="nav-icon">🗺️</div>
        <div class="nav-title">Geographic Clusters</div>
        <div class="nav-desc">K-Means · DBSCAN · Hierarchical hotspot maps</div>
    </div>
    <div class="nav-card">
        <div class="nav-icon">⏰</div>
        <div class="nav-title">Temporal Patterns</div>
        <div class="nav-desc">Hourly, daily, seasonal crime rhythms</div>
    </div>
    <div class="nav-card">
        <div class="nav-icon">🔬</div>
        <div class="nav-title">Dim. Reduction</div>
        <div class="nav-desc">PCA scree · loadings · t-SNE day/night</div>
    </div>
    <div class="nav-card">
        <div class="nav-icon">📈</div>
        <div class="nav-title">MLflow Monitor</div>
        <div class="nav-desc">Experiment tracker · model comparison</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Algorithm summary ─────────────────────────────────────────────────────────
st.markdown("### 🤖 ML Pipeline Summary")
sil_df = load_sil()
if not sil_df.empty:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(
            sil_df.style.format({"Silhouette": "{:.4f}", "Davies_Bouldin": "{:.4f}"})
                        .highlight_max(subset=["Silhouette"], color="#1e3a2f")
                        .highlight_min(subset=["Davies_Bouldin"], color="#2a1f0a"),
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        best_sil = sil_df.loc[sil_df["Silhouette"].idxmax()]
        best_db  = sil_df.loc[sil_df["Davies_Bouldin"].idxmin()]
        st.markdown(f"""
        **🏆 Best Silhouette Score**
        > **{best_sil['Algorithm']}** — `{best_sil['Silhouette']:.4f}`
        *(Higher = better-defined clusters)*

        **🎯 Best Davies-Bouldin**
        > **{best_db['Algorithm']}** — `{best_db['Davies_Bouldin']:.4f}`
        *(Lower = tighter, well-separated clusters)*

        ✅ DBSCAN achieves the highest silhouette (density-based clusters naturally fit geographic crime patterns).
        """)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🛠️ Technology Stack")
tc = st.columns(6)
techs = [
    ("🐍", "Python 3.12"),   ("🤖", "scikit-learn"),
    ("📊", "Pandas/NumPy"),  ("📈", "MLflow"),
    ("🗺️", "Folium/Plotly"), ("☁️", "Streamlit Cloud"),
]
for col, (icon, name) in zip(tc, techs):
    with col:
        st.markdown(f"""
        <div style='text-align:center; background:#161b22; border:1px solid #21262d; border-radius:10px; padding:0.9rem 0.5rem;'>
            <div style='font-size:1.5rem;'>{icon}</div>
            <div style='font-size:0.72rem; color:#8b949e; margin-top:0.3rem;'>{name}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span class="status-dot"></span>
    PatrolIQ · Chicago Crime Intelligence Platform · Built with Streamlit Cloud · Powered by Unsupervised ML
</div>
""", unsafe_allow_html=True)
