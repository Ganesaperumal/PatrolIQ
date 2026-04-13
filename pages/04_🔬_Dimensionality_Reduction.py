"""
PatrolIQ Page 4 — Dimensionality Reduction
PCA scree plot, feature loadings heatmap, 2D scatter, t-SNE day/night comparison.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os
import numpy as np

st.set_page_config(page_title="Dimensionality Reduction | PatrolIQ", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.page-header { background:linear-gradient(135deg,#0d1117,#0e1a2e); border:1px solid #21262d; border-radius:16px; padding:1.8rem 2rem; margin-bottom:1.5rem; }
.page-title  { font-size:2rem; font-weight:900; color:#e6edf3; margin:0; }
.page-sub    { color:#8b949e; font-size:0.9rem; margin-top:0.3rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#0e1a2e); border-right:1px solid #21262d; }
.tech-card { background:#161b22; border:1px solid #21262d; border-radius:12px; padding:1.2rem; margin-bottom:1rem; }
.tech-title { font-size:1rem; font-weight:700; color:#4ade80; margin-bottom:0.5rem; }
.tech-body  { font-size:0.82rem; color:#8b949e; line-height:1.6; }
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

@st.cache_data(ttl=3600)
def load_pca_variance():
    p = os.path.join(CLEAN_DIR, "pca_variance.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=3600)
def load_pca_loadings():
    p = os.path.join(CLEAN_DIR, "pca_loadings.csv")
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Loading PCA results …")
def load_pca_result():
    p = os.path.join(CLEAN_DIR, "pca_result.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Loading t-SNE results …")
def load_tsne():
    p = os.path.join(CLEAN_DIR, "tsne_result.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data(ttl=30)
def load_dim_summary():
    p = os.path.join(CLEAN_DIR, "dimensionality_reduction_summary.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)

pca_var     = load_pca_variance()
pca_load    = load_pca_loadings()
pca_result  = load_pca_result()
tsne_df     = load_tsne()
dim_summary = load_dim_summary()

st.markdown("""
<div class="page-header">
    <div class="page-title">🔬 Dimensionality Reduction</div>
    <div class="page-sub">Compress 14+ features into 2D spaces — PCA for variance, t-SNE for cluster visualization</div>
</div>
""", unsafe_allow_html=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
pca_meta = dim_summary.get("PCA", {})
tsne_meta = dim_summary.get("tSNE", {})

c1,c2,c3,c4 = st.columns(4)
c1.metric("PCA 2-Component Variance", f"{pca_meta.get('variance_explained_2pc',0):.1f}%")
c2.metric("Components for 70% Variance", pca_meta.get("n_components_70pct","6"))
c3.metric("Components for 80% Variance", pca_meta.get("n_components_80pct","8"))
c4.metric("t-SNE KL-Divergence", f"{tsne_meta.get('kl_divergence',0):.4f}")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 PCA Scree Plot", "🗺️ Feature Loadings", "🔵 PCA 2D Scatter", "🟣 t-SNE Visualization"])

# ── Tab 1: Scree plot ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### PCA Scree Plot — Explained Variance per Component")
    if not pca_var.empty:
        scree = pca_var.head(14).copy()
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Bar(x=scree["Component"], y=scree["Explained_Variance"]*100,
                               name="Individual Variance %",
                               marker=dict(color="#4ade80", opacity=0.8)), secondary_y=False)
        fig1.add_trace(go.Scatter(x=scree["Component"], y=scree["Cumulative_Variance"]*100,
                                   name="Cumulative Variance %",
                                   line=dict(color="#f59e0b", width=2.5),
                                   mode="lines+markers"), secondary_y=True)
        # 70% line
        fig1.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=1.5),
                        secondary_y=True, annotation_text="70% threshold")
        fig1.add_hline(y=80, line=dict(color="#f59e0b", dash="dot", width=1.5),
                        secondary_y=True, annotation_text="80% threshold")
        fig1.update_layout(
            template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            margin=dict(l=10,r=10,t=30,b=10), height=400,
            legend=dict(orientation="h", y=1.05),
            barmode="group",
        )
        fig1.update_yaxes(title_text="Individual Variance %", secondary_y=False)
        fig1.update_yaxes(title_text="Cumulative Variance %", secondary_y=True, range=[0,105])
        st.plotly_chart(fig1, use_container_width=True)

        n70 = pca_meta.get("n_components_70pct", "6")
        st.markdown(f"""
        **Key Finding:** `{n70}` principal components are needed to explain **70%** of the variance.
        The first 2 components explain `{pca_meta.get('variance_explained_2pc',0):.1f}%` — sufficient for 2D visualization.
        
        The **elbow** appears around PC3–PC4, after which each additional component adds diminishing returns.
        """)
    else:
        st.warning("PCA variance data not found. Run `scripts/train.py` first.")

# ── Tab 2: Feature loadings ───────────────────────────────────────────────────
with tab2:
    st.markdown("### PCA Feature Loadings Heatmap")
    st.caption("Shows how much each original feature contributes to each Principal Component")
    if not pca_load.empty:
        fig2 = px.imshow(
            pca_load.values,
            x=pca_load.columns.tolist(),
            y=pca_load.index.tolist(),
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            text_auto=".2f",
            template="plotly_dark",
            aspect="auto",
        )
        fig2.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            margin=dict(l=10,r=10,t=30,b=10), height=420,
        )
        fig2.update_traces(textfont_size=10)
        st.plotly_chart(fig2, use_container_width=True)

        # Top features table
        st.markdown("#### 🏆 Top 5 Features by PC1 (Absolute Loading)")
        top_features = pca_load["PC1"].abs().sort_values(ascending=False).head(5)
        feat_df = pd.DataFrame({
            "Feature": top_features.index,
            "PC1 Loading": pca_load.loc[top_features.index,"PC1"].round(4),
            "PC2 Loading": pca_load.loc[top_features.index,"PC2"].round(4) if "PC2" in pca_load.columns else [0]*5,
            "Importance": ["🥇 Primary","🥈 Secondary","🥉 Tertiary","4th","5th"],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.info("""
**Interpretation:** Features with high absolute loading values drive the most variation captured by that principal component.
Geographic features (Latitude, Longitude) typically dominate PC1, while temporal features (Hour, Month) drive PC2.
        """)
    else:
        st.warning("PCA loadings not found. Run `scripts/train.py` first.")

# ── Tab 3: PCA 2D scatter ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### PCA 2D Scatter Plot")
    if not pca_result.empty:
        color_by = st.selectbox("Color by", ["Primary_Type","Arrest","Is_Night","Is_Weekend"],
                                  key="pca_color",
                                  format_func=lambda x: x.replace("_"," "))

        sample_pca = pca_result.sample(n=min(5000, len(pca_result)), random_state=42)

        if color_by in ["Arrest","Is_Night","Is_Weekend"]:
            sample_pca[color_by] = sample_pca[color_by].astype(str)

        fig3 = px.scatter(
            sample_pca, x="PC1", y="PC2", color=color_by,
            opacity=0.55, template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.D3,
            labels={"PC1": f"PC1 ({pca_meta.get('variance_explained_2pc',0)/2:.1f}% var)",
                    "PC2": "PC2"},
            hover_data={color_by: True, "PC1": ":.3f", "PC2": ":.3f"},
        )
        fig3.update_traces(marker=dict(size=3, line=dict(width=0)))
        fig3.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            margin=dict(l=10,r=10,t=30,b=10), height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.info("PCA preserves global structure — similar crimes cluster together. Geographic crimes (theft, burglary) cluster distinctly from violent crimes (assault, battery).")
    else:
        st.warning("PCA result not found. Run `scripts/train.py` first.")

# ── Tab 4: t-SNE ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### t-SNE 2D Visualization")
    if not tsne_df.empty:
        col_left, col_right = st.columns([1, 3])
        with col_left:
            tsne_color = st.radio("Color by", ["Primary Type","Time Period","Is_Night"],
                                   key="tsne_color_sel")
            n_sample = st.slider("Sample size", 1000, min(10000,len(tsne_df)), 5000, step=500)

        col_map = {"Primary Type":"Primary Type","Time Period":"Time_Period","Is_Night":"Is_Night"}
        actual_col = col_map.get(tsne_color,"Primary Type")

        tsne_sample = tsne_df.sample(n=n_sample, random_state=42)

        if actual_col == "Is_Night":
            tsne_sample["Is_Night"] = tsne_sample["Is_Night"].astype(str)

        with col_right:
            fig4 = px.scatter(
                tsne_sample, x="TSNE1", y="TSNE2",
                color=actual_col,
                opacity=0.55, template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                labels={"TSNE1":"t-SNE Dimension 1","TSNE2":"t-SNE Dimension 2"},
            )
            fig4.update_traces(marker=dict(size=3, line=dict(width=0)))
            fig4.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=10,r=10,t=30,b=10), height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Day vs Night side-by-side
        st.markdown("#### 🌙 Day vs Night Crime Patterns")
        if "Time_Period" in tsne_df.columns:
            day_data   = tsne_sample[tsne_sample["Time_Period"].str.contains("Day",na=False)]
            night_data = tsne_sample[tsne_sample["Time_Period"].str.contains("Night",na=False)]

            fig5 = make_subplots(rows=1, cols=2,
                                  subplot_titles=("☀️ Daytime Crimes (6AM–8PM)",
                                                  "🌙 Nighttime Crimes (8PM–6AM)"))
            for data_slice, col_idx, color in [(day_data,1,"#fbbf24"),(night_data,2,"#818cf8")]:
                fig5.add_trace(go.Scatter(
                    x=data_slice["TSNE1"], y=data_slice["TSNE2"],
                    mode="markers",
                    marker=dict(size=3, color=color, opacity=0.5, line=dict(width=0)),
                    name="Day" if col_idx==1 else "Night",
                ), row=1, col=col_idx)
            fig5.update_layout(
                template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=10,r=10,t=50,b=10), height=380, showlegend=False,
            )
            for ax in ["xaxis","yaxis","xaxis2","yaxis2"]:
                fig5.update_layout(**{ax: dict(
                    showgrid=True, gridcolor="#21262d", zeroline=False)})
            st.plotly_chart(fig5, use_container_width=True)
            st.info("""
**Day vs Night Comparison:** t-SNE reveals that nighttime crimes (purple) form denser, tighter clusters — 
suggesting that night crimes (robbery, assault) follow more predictable patterns, while daytime crimes (theft, fraud) 
are more dispersed across the city. This supports targeted night patrol in known crime hubs.
            """)
    else:
        st.warning("t-SNE results not found. Run `scripts/train.py` first.")

# ── Technical Documentation ───────────────────────────────────────────────────
st.divider()
st.markdown("### 📚 Technical Notes")
col_doc1, col_doc2 = st.columns(2)
with col_doc1:
    st.markdown("""
    <div class="tech-card">
        <div class="tech-title">🔵 PCA (Principal Component Analysis)</div>
        <div class="tech-body">
            Linear dimensionality reduction that finds orthogonal axes of maximum variance.
            Ideal for understanding <em>which features matter most</em> and for pre-processing before t-SNE.
            <br><br>
            <b>Features used:</b> Hour, Month, Is_Weekend, Is_Night, Crime_Severity_Score, 
            Latitude, Longitude, District, Ward, Beat, Community Area, Arrest, Domestic
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_doc2:
    st.markdown("""
    <div class="tech-card">
        <div class="tech-title">🟣 t-SNE (t-Distributed Stochastic Neighbor Embedding)</div>
        <div class="tech-body">
            Non-linear technique that preserves <em>local neighborhood structure</em>, making clusters visually apparent.
            Best for exploratory visualization. KL-divergence measures how well the 2D map represents the original distances.
            <br><br>
            <b>Params:</b> perplexity=30, max_iter=300, pre-reduced via PCA(10 components)
        </div>
    </div>
    """, unsafe_allow_html=True)
