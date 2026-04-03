"""
PatrolIQ - Dimensionality Reduction Page
PCA scree + 3D scatter, t-SNE 2D, UMAP 2D with color-coded clusters
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Dimensionality Reduction — PatrolIQ", layout="wide", page_icon="🔬")
BASE_DIR = Path(__file__).resolve().parents[2]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
html,body,.stApp{background:#0a0e1a;}
.page-title{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#f093fb,#f5576c);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem;}
.sub-text{color:#8892b0;margin-bottom:2rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#161b2e);border-right:1px solid #2a3050;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔬 Dimensionality Reduction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Compressing 40+ features into 2D/3D visual space to uncover hidden crime patterns</div>', unsafe_allow_html=True)

SUM_DIR = BASE_DIR / "reports" / "summaries"
FIG_DIR = BASE_DIR / "reports" / "figures"

tab1, tab2, tab3 = st.tabs(["📐 PCA", "🌀 t-SNE", "🔵 UMAP"])

# ── COLOR COLUMN SELECTOR ─────────────────────────────────
model_ready = BASE_DIR / "data" / "processed" / "model_ready_data.csv"
@st.cache_data
def load_labels():
    if not model_ready.exists(): return None
    df = pd.read_csv(model_ready, low_memory=False)
    sample = df.sample(min(10_000, len(df)), random_state=42)
    return sample

labels_df = load_labels()
color_options = []
if labels_df is not None:
    for c in ["Crime_Severity_Score","Time_Label","Crime_Label","Season_Label","Is_Weekend"]:
        if c in labels_df.columns: color_options.append(c)

with st.sidebar:
    st.markdown("### 🎨 Color Encoding")
    color_by = st.selectbox("Color points by", color_options if color_options else ["N/A"])

def dark_layout(fig):
    fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
    return fig

# ── PCA TAB ───────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        # Scree plot
        scree_img = FIG_DIR / "pca_scree_plot.png"
        if scree_img.exists():
            st.image(str(scree_img), caption="PCA Scree Plot — Cumulative Variance Explained", use_container_width=True)
        else:
            st.info("Run `dimensionality_reduction.py` to generate the scree plot.")

    with col2:
        # PCA summary
        import json
        summary_path = SUM_DIR / "dimensionality_reduction_summary.json"
        if summary_path.exists():
            with open(summary_path) as f: summary = json.load(f)
            st.markdown("#### 📊 PCA Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("PC1 Variance", f"{summary.get('PC1_var', 0):.1%}")
            c2.metric("PC2 Variance", f"{summary.get('PC2_var', 0):.1%}")
            c3.metric("PC3 Variance", f"{summary.get('PC3_var', 0):.1%}")
            st.metric("Total Variance (3 PCs)", f"{summary.get('total_explained_variance', 0):.1%}")

    # 3D PCA scatter
    pca_path = SUM_DIR / "pca_reduced_data.csv"
    if pca_path.exists():
        pca_df = pd.read_csv(pca_path)
        sample_pca = pca_df.sample(min(5000, len(pca_df)), random_state=42).reset_index(drop=True)
        # Merge color column if available
        if labels_df is not None and color_by in labels_df.columns:
            sample_pca[color_by] = labels_df[color_by].values[:len(sample_pca)]
        color_col = color_by if color_by in sample_pca.columns else None
        fig_pca3d = px.scatter_3d(sample_pca, x="PCA1", y="PCA2", z="PCA3",
                                  color=color_col, opacity=0.7,
                                  title="PCA 3D Crime Pattern Space",
                                  color_continuous_scale="Viridis")
        fig_pca3d.update_traces(marker_size=2)
        fig_pca3d.update_layout(paper_bgcolor="#0a0e1a", font_color="#ccd6f6", height=550)
        st.plotly_chart(fig_pca3d, use_container_width=True)

        # Feature loadings
        loadings_path = SUM_DIR / "pca_feature_loadings.csv"
        if loadings_path.exists():
            with st.expander("📋 Top Feature Loadings (PC1)"):
                loadings_df = pd.read_csv(loadings_path, index_col=0)
                loadings_df["PC1_abs"] = loadings_df["PC1"].abs()
                st.dataframe(loadings_df.sort_values("PC1_abs", ascending=False).head(10).drop("PC1_abs", axis=1),
                             use_container_width=True)
    else:
        st.warning("Run `dimensionality_reduction.py` to generate PCA results.")

# ── t-SNE TAB ─────────────────────────────────────────────
with tab2:
    tsne_path = SUM_DIR / "tsne_reduced_data.csv"
    if tsne_path.exists():
        tsne_df = pd.read_csv(tsne_path).reset_index(drop=True)
        if labels_df is not None and color_by in labels_df.columns:
            n = min(len(tsne_df), len(labels_df))
            tsne_df[color_by] = labels_df[color_by].values[:n]
        color_col = color_by if color_by in tsne_df.columns else None
        fig_tsne = px.scatter(tsne_df.sample(min(8000,len(tsne_df)),random_state=42),
                              x="TSNE1", y="TSNE2", color=color_col, opacity=0.65,
                              title="t-SNE Crime Pattern Visualization (Sample 10k)",
                              color_continuous_scale="Plasma")
        dark_layout(fig_tsne)
        st.plotly_chart(fig_tsne, use_container_width=True)
        st.caption("t-SNE reveals non-linear patterns. Spatial proximity = behavioral similarity between crimes.")
    else:
        st.warning("Run `dimensionality_reduction.py` to generate t-SNE results.")

# ── UMAP TAB ──────────────────────────────────────────────
with tab3:
    umap_path = SUM_DIR / "umap_reduced_data.csv"
    if umap_path.exists():
        umap_df = pd.read_csv(umap_path).reset_index(drop=True)
        if labels_df is not None and color_by in labels_df.columns:
            n = min(len(umap_df), len(labels_df))
            umap_df[color_by] = labels_df[color_by].values[:n]
        color_col = color_by if color_by in umap_df.columns else None
        fig_umap = px.scatter(umap_df.sample(min(10000,len(umap_df)),random_state=42),
                              x="UMAP1", y="UMAP2", color=color_col, opacity=0.6,
                              title="UMAP Crime Pattern Visualization (Sample 20k)",
                              color_continuous_scale="Inferno")
        dark_layout(fig_umap)
        st.plotly_chart(fig_umap, use_container_width=True)
        st.caption("UMAP preserves both local and global structure better than t-SNE for large datasets.")
    else:
        st.warning("Run `dimensionality_reduction.py` to generate UMAP results. (Requires umap-learn)")
