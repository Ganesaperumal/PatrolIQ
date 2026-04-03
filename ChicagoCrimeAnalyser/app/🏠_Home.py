"""
PatrolIQ - Premium Home Page
Best-of-both: senior's premium CSS animations + user's dark theme
"""
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PatrolIQ — Crime Intelligence Platform",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, .stApp { background: #0a0e1a; font-family: 'Inter', sans-serif; }
.main .block-container { padding: 2rem 3rem; max-width: 1600px; }

/* ── Hero ── */
.hero { text-align:center; padding: 3rem 0 2rem; }
.hero-badge {
    display:inline-block; background:linear-gradient(135deg,#667eea,#764ba2);
    color:#fff; padding:.5rem 1.5rem; border-radius:50px; font-size:.85rem;
    font-weight:700; letter-spacing:2px; text-transform:uppercase;
    margin-bottom:1.5rem; animation:float 3s ease-in-out infinite;
}
.hero-title {
    font-family:'Outfit',sans-serif; font-size:5.5rem; font-weight:900;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 30%,#f093fb 60%,#4facfe 100%);
    background-size:400% 400%; -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; background-clip:text;
    animation:gradient 8s ease infinite; letter-spacing:-3px; line-height:1;
    margin-bottom:1rem;
}
.hero-sub { font-size:1.2rem; color:#8892b0; font-weight:400; max-width:600px; margin:0 auto 2rem; line-height:1.8; }

/* ── Animations ── */
@keyframes gradient { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes slideUp { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(102,126,234,0.4)} 50%{box-shadow:0 0 0 15px rgba(102,126,234,0)} }

/* ── Metric Cards ── */
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; margin:2rem 0; }
.metric-card {
    background:linear-gradient(135deg,#161b2e,#1a2036); border:1px solid #2a3050;
    border-radius:20px; padding:2rem 1.5rem; text-align:center;
    transition:all .4s cubic-bezier(.4,0,.2,1); animation:slideUp .6s ease-out;
    position:relative; overflow:hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#667eea,#764ba2,#f093fb);
    transform:scaleX(0); transform-origin:left; transition:transform .4s ease;
}
.metric-card:hover::before { transform:scaleX(1); }
.metric-card:hover { transform:translateY(-8px); border-color:#667eea; box-shadow:0 20px 40px rgba(102,126,234,.2); }
.metric-num { font-family:'Outfit',sans-serif; font-size:3.2rem; font-weight:900; background:linear-gradient(135deg,#667eea,#4facfe); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1; }
.metric-lbl { color:#8892b0; font-size:.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1.5px; margin-top:.5rem; }

/* ── Feature Cards ── */
.feature-card {
    background:linear-gradient(135deg,#161b2e,#1a2036); border:1px solid #2a3050;
    border-radius:16px; padding:2rem; transition:all .3s ease; margin-bottom:1rem;
}
.feature-card:hover { transform:translateY(-5px); border-color:#667eea; box-shadow:0 15px 35px rgba(102,126,234,.15); }
.feature-icon { font-size:2.5rem; margin-bottom:1rem; }
.feature-title { color:#ccd6f6; font-size:1.2rem; font-weight:700; font-family:'Outfit',sans-serif; margin-bottom:.5rem; }
.feature-desc { color:#8892b0; font-size:.9rem; line-height:1.7; }
.feature-tags { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }
.tag { background:rgba(102,126,234,.15); color:#667eea; border:1px solid rgba(102,126,234,.3); border-radius:20px; padding:.25rem .75rem; font-size:.75rem; font-weight:600; }

/* ── Nav Guide ── */
.nav-item { background:linear-gradient(135deg,#161b2e,#1a2036); border:1px solid #2a3050; border-radius:14px; padding:1.2rem 1.5rem; margin:.5rem 0; display:flex; align-items:center; gap:1rem; transition:all .3s; }
.nav-item:hover { border-color:#667eea; transform:translateX(8px); background:linear-gradient(135deg,#1a2036,#1e2540); }
.nav-num { background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.85rem; flex-shrink:0; }
.nav-text { color:#ccd6f6; font-size:.95rem; font-weight:500; }

/* ── Divider ── */
.gradient-divider { height:2px; background:linear-gradient(90deg,transparent,#667eea,#764ba2,transparent); border:none; margin:2.5rem 0; }

/* ── Section title ── */
.section-title { font-family:'Outfit',sans-serif; font-size:2rem; font-weight:800; color:#ccd6f6; margin-bottom:1.5rem; }
.section-title span { background:linear-gradient(135deg,#667eea,#f093fb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#161b2e); border-right:1px solid #2a3050; }
[data-testid="stSidebar"] .stRadio label { color:#8892b0 !important; }

/* ── Footer ── */
.footer { text-align:center; color:#4a5568; padding:2rem 0; margin-top:3rem; border-top:1px solid #1e2540; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">🚔 Chicago Crime Intelligence</div>
    <div class="hero-title">PatrolIQ</div>
    <p class="hero-sub">AI-Powered Urban Safety Analytics Platform<br>Transforming 500,000 crime records into actionable intelligence</p>
</div>
""", unsafe_allow_html=True)

# Metric cards
st.markdown("""
<div class="metric-grid">
    <div class="metric-card"><div class="metric-num">500K</div><div class="metric-lbl">Crime Records</div></div>
    <div class="metric-card"><div class="metric-num">33</div><div class="metric-lbl">Crime Types</div></div>
    <div class="metric-card"><div class="metric-num">9</div><div class="metric-lbl">Hotspot Zones</div></div>
    <div class="metric-card"><div class="metric-num">4</div><div class="metric-lbl">Time Patterns</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

# Feature Cards
st.markdown('<div class="section-title">🚀 Platform <span>Capabilities</span></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🗺️</div>
        <div class="feature-title">Geographic Intelligence</div>
        <div class="feature-desc">Spatial clustering identifies 9 crime hotspot zones using K-Means, DBSCAN, and Hierarchical algorithms with interactive Folium maps.</div>
        <div class="feature-tags"><span class="tag">K-Means</span><span class="tag">DBSCAN</span><span class="tag">Hierarchical</span></div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⏰</div>
        <div class="feature-title">Temporal Analysis</div>
        <div class="feature-desc">Discover when crimes peak using cyclical time encoding, weekday-hour heatmaps, and seasonal pattern analysis across all 24 hours.</div>
        <div class="feature-tags"><span class="tag">Cyclic Encoding</span><span class="tag">Seasonal</span><span class="tag">Hourly</span></div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Dimensionality Reduction</div>
        <div class="feature-desc">Compress 40+ features into stunning 3D PCA, t-SNE, and UMAP visualizations to uncover hidden crime patterns and behavioral clusters.</div>
        <div class="feature-tags"><span class="tag">PCA</span><span class="tag">t-SNE</span><span class="tag">UMAP</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)

# Navigation Guide
st.markdown('<div class="section-title">🎯 <span>Navigation</span> Guide</div>', unsafe_allow_html=True)
for i, (emoji, page, desc) in enumerate([
    ("🎯","Clustering Analysis","Compare K-Means, DBSCAN & Hierarchical with Silhouette + Davies-Bouldin scores"),
    ("⏰","Temporal Analysis","Explore hourly, weekday, seasonal crime patterns and time-based clusters"),
    ("🔬","Dimensionality Reduction","Interact with 3D PCA, t-SNE and UMAP crime visualizations"),
    ("📊","EDA Insights","Browse crime type rankings, arrest rates, and full dataset statistics"),
    ("🗺️","Geographic Heatmaps","Switch between clustering algorithms on an interactive Chicago map"),
    ("📈","MLflow Monitoring","Track all experiment runs, parameter sweeps, and model metrics"),
], start=1):
    st.markdown(f"""
    <div class="nav-item">
        <div class="nav-num">{i}</div>
        <div class="nav-text"><strong style="color:#ccd6f6">{emoji} {page}</strong> — {desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    PatrolIQ Crime Intelligence Platform &nbsp;•&nbsp;
    Powered by Streamlit · Scikit-learn · MLflow · Plotly · Folium<br>
    Chicago Crime Data (2010–2025) · Built for Public Safety
</div>
""", unsafe_allow_html=True)
