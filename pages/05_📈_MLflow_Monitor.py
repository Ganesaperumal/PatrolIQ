"""
PatrolIQ Page 5 — MLflow Monitor
Experiment tracking dashboard: all runs, metrics, best model selection.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="MLflow Monitor | PatrolIQ", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.page-header { background:linear-gradient(135deg,#0d1117,#0e1a2e); border:1px solid #21262d; border-radius:16px; padding:1.8rem 2rem; margin-bottom:1.5rem; }
.page-title  { font-size:2rem; font-weight:900; color:#e6edf3; margin:0; }
.page-sub    { color:#8b949e; font-size:0.9rem; margin-top:0.3rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#0e1a2e); border-right:1px solid #21262d; }
.run-card { background:#161b22; border:1px solid #21262d; border-radius:12px; padding:1rem; margin-bottom:0.7rem; }
.run-card:hover { border-color:#4ade80; }
.run-name { font-weight:700; color:#e6edf3; font-size:0.95rem; }
.run-metric { font-size:0.8rem; color:#8b949e; margin-top:0.3rem; }
.badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.7rem; font-weight:600; margin-left:0.5rem; }
.badge-geo  { background:rgba(74,222,128,0.15); color:#4ade80; }
.badge-temp { background:rgba(96,165,250,0.15); color:#60a5fa; }
.badge-dim  { background:rgba(167,139,250,0.15); color:#a78bfa; }
.winner-box { background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.3); border-radius:12px; padding:1.2rem 1.5rem; }
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

CLEAN_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
MLFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")

# ── Load metrics from JSON files ───────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_all_metrics():
    rows = []
    # Geo clustering
    gp = os.path.join(CLEAN_DIR, "geo_clustering_metrics.json")
    if os.path.exists(gp):
        geo = json.load(open(gp))
        for name, m in geo.items():
            rows.append({
                "Run Name": name, "Type": "Geographic",
                "Silhouette": m.get("silhouette",0),
                "Davies_Bouldin": m.get("davies_bouldin",0),
                "N_Clusters": m.get("n_clusters","N/A"),
                "Noise %": m.get("noise_pct","—"),
                "Status": "✅ Complete",
            })
    # Temporal
    tp = os.path.join(CLEAN_DIR, "temporal_clustering_metrics.json")
    if os.path.exists(tp):
        temp = json.load(open(tp))
        for name, m in temp.items():
            rows.append({
                "Run Name": name, "Type": "Temporal",
                "Silhouette": m.get("silhouette",0),
                "Davies_Bouldin": m.get("davies_bouldin",0),
                "N_Clusters": m.get("n_clusters","N/A"),
                "Noise %": "—",
                "Status": "✅ Complete",
            })
    # Dimensionality reduction
    dp = os.path.join(CLEAN_DIR, "dimensionality_reduction_summary.json")
    if os.path.exists(dp):
        dim = json.load(open(dp))
        if "PCA" in dim:
            rows.append({
                "Run Name": "PCA_DimensionalityReduction", "Type": "Dim. Reduction",
                "Silhouette": "—",
                "Davies_Bouldin": "—",
                "N_Clusters": dim["PCA"].get("n_components_70pct","—"),
                "Noise %": f"{dim['PCA'].get('variance_explained_2pc',0):.1f}% var (2PC)",
                "Status": "✅ Complete",
            })
        if "tSNE" in dim:
            rows.append({
                "Run Name": "tSNE_Visualization", "Type": "Dim. Reduction",
                "Silhouette": f"KL={dim['tSNE'].get('kl_divergence',0):.4f}",
                "Davies_Bouldin": "—",
                "N_Clusters": "2D",
                "Noise %": "—",
                "Status": "✅ Complete",
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=60)
def load_sil():
    p = os.path.join(CLEAN_DIR, "silhouette_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

runs_df = load_all_metrics()
sil_df  = load_sil()

st.markdown("""
<div class="page-header">
    <div class="page-title">📈 MLflow Experiment Monitor</div>
    <div class="page-sub">Track all experiment runs, compare model performance, and select the best algorithm</div>
</div>
""", unsafe_allow_html=True)

# ── Summary KPIs ─────────────────────────────────────────────────────────────
total_runs = len(runs_df)
geo_runs   = len(runs_df[runs_df["Type"]=="Geographic"]) if not runs_df.empty else 0
temp_runs  = len(runs_df[runs_df["Type"]=="Temporal"]) if not runs_df.empty else 0
dim_runs   = len(runs_df[runs_df["Type"]=="Dim. Reduction"]) if not runs_df.empty else 0

c1,c2,c3,c4 = st.columns(4)
for col, val, lbl, icon in [
    (c1, total_runs, "Total MLflow Runs", "🧪"),
    (c2, geo_runs,   "Geographic Models",  "🗺️"),
    (c3, temp_runs,  "Temporal Models",    "⏰"),
    (c4, dim_runs,   "Dim. Reduction Runs","🔬"),
]:
    with col:
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #21262d;border-radius:12px;padding:1rem;text-align:center;'>
            <div style='font-size:1.5rem'>{icon}</div>
            <div style='font-size:1.8rem;font-weight:800;color:#4ade80'>{val}</div>
            <div style='font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 All Runs", "📊 Performance Charts", "🎯 Radar Comparison",
    "🏆 Best Model", "📁 Experiment Details"
])

# ── Tab 1: All runs ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### All Experiment Runs")
    if not runs_df.empty:
        st.dataframe(
            runs_df.style.applymap(
                lambda v: "color: #4ade80" if v == "✅ Complete" else "", subset=["Status"]
            ),
            use_container_width=True, hide_index=True, height=350,
        )
    else:
        st.warning("No MLflow runs found. Run `scripts/train.py` first.")

    st.markdown("### Experiment: `PatrolIQ_Crime_Analysis`")
    mlruns_path = MLFLOW_DIR
    if os.path.exists(mlruns_path):
        st.success(f"✅ MLflow tracking directory found: `{mlruns_path}`")
        # Count experiments
        exp_dirs = [d for d in os.listdir(mlruns_path) if os.path.isdir(os.path.join(mlruns_path,d))]
        st.info(f"Found **{len(exp_dirs)}** experiment folder(s) in mlruns.")
    else:
        st.error("MLflow directory not found. Run `scripts/train.py` first.")

# ── Tab 2: Performance charts ─────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Performance Comparison")
    if not sil_df.empty:
        geo_sil = sil_df[sil_df["Algorithm"].str.contains("Geo|Temporal", na=False)]

        col_a, col_b = st.columns(2)
        with col_a:
            fig1 = px.bar(geo_sil, x="Algorithm", y="Silhouette",
                           color="Silhouette", color_continuous_scale="Teal",
                           template="plotly_dark",
                           title="Silhouette Score (↑ Better)")
            fig1.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                coloraxis_showscale=False, height=320,
                                margin=dict(l=10,r=10,t=40,b=10))
            fig1.add_hline(y=0.5, line=dict(color="#ef4444", dash="dash"),
                            annotation_text="Target: 0.5")
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            geo_sil_valid = geo_sil[pd.to_numeric(geo_sil["Davies_Bouldin"], errors="coerce").notna()]
            geo_sil_valid = geo_sil_valid.copy()
            geo_sil_valid["Davies_Bouldin"] = pd.to_numeric(geo_sil_valid["Davies_Bouldin"])
            fig2 = px.bar(geo_sil_valid, x="Algorithm", y="Davies_Bouldin",
                           color="Davies_Bouldin", color_continuous_scale="Reds_r",
                           template="plotly_dark",
                           title="Davies-Bouldin Index (↓ Better)")
            fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                coloraxis_showscale=False, height=320,
                                margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # Combined scatter
        st.markdown("#### Efficiency Frontier: Silhouette vs Davies-Bouldin")
        geo_both = geo_sil_valid.copy()
        fig3 = px.scatter(geo_both, x="Davies_Bouldin", y="Silhouette",
                           text="Algorithm", size_max=20,
                           template="plotly_dark",
                           color="Algorithm",
                           color_discrete_sequence=["#4ade80","#60a5fa","#f59e0b","#f87171"])
        fig3.update_traces(textposition="top center", marker_size=14)
        fig3.add_annotation(text="Ideal: Low DB, High Silhouette →",
                             xref="paper", yref="paper", x=0.98, y=0.1,
                             showarrow=False, font=dict(color="#8b949e", size=10))
        fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                            margin=dict(l=10,r=10,t=30,b=10), height=350,
                            showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Metrics not found. Run `scripts/train.py` first.")

# ── Tab 3: Radar chart ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Multi-Metric Radar Comparison")
    if not sil_df.empty:
        geo_sil_r = sil_df[sil_df["Algorithm"].str.contains("Geo",na=False)].copy()
        geo_sil_r["Silhouette_n"] = pd.to_numeric(geo_sil_r["Silhouette"], errors="coerce").fillna(0)
        geo_sil_r["DB_n"] = pd.to_numeric(geo_sil_r["Davies_Bouldin"], errors="coerce").fillna(0)

        max_sil = geo_sil_r["Silhouette_n"].max()
        max_db  = geo_sil_r["DB_n"].max()

        # Manual scoring for qualitative criteria
        algo_scores = {
            "KMeans_Geo":       {"Silhouette":geo_sil_r[geo_sil_r["Algorithm"]=="KMeans_Geo"]["Silhouette_n"].values[0] if len(geo_sil_r[geo_sil_r["Algorithm"]=="KMeans_Geo"]) > 0 else 0,
                                  "DB_inv":1-(geo_sil_r[geo_sil_r["Algorithm"]=="KMeans_Geo"]["DB_n"].values[0]/max_db) if len(geo_sil_r[geo_sil_r["Algorithm"]=="KMeans_Geo"]) > 0 else 0,
                                  "Speed":0.95,"Scalability":0.92,"Interpretability":0.90},
            "DBSCAN_Geo":       {"Silhouette":geo_sil_r[geo_sil_r["Algorithm"]=="DBSCAN_Geo"]["Silhouette_n"].values[0] if len(geo_sil_r[geo_sil_r["Algorithm"]=="DBSCAN_Geo"]) > 0 else 0,
                                  "DB_inv":1-(geo_sil_r[geo_sil_r["Algorithm"]=="DBSCAN_Geo"]["DB_n"].values[0]/max_db) if len(geo_sil_r[geo_sil_r["Algorithm"]=="DBSCAN_Geo"]) > 0 else 0,
                                  "Speed":0.70,"Scalability":0.65,"Interpretability":0.72},
            "Hierarchical_Geo":{"Silhouette":geo_sil_r[geo_sil_r["Algorithm"]=="Hierarchical_Geo"]["Silhouette_n"].values[0] if len(geo_sil_r[geo_sil_r["Algorithm"]=="Hierarchical_Geo"]) > 0 else 0,
                                  "DB_inv":1-(geo_sil_r[geo_sil_r["Algorithm"]=="Hierarchical_Geo"]["DB_n"].values[0]/max_db) if len(geo_sil_r[geo_sil_r["Algorithm"]=="Hierarchical_Geo"]) > 0 else 0,
                                  "Speed":0.55,"Scalability":0.50,"Interpretability":0.85},
        }

        cats = ["Silhouette", "DB (inv.)", "Speed", "Scalability", "Interpretability"]
        colors_r = {"KMeans_Geo":"#4ade80","DBSCAN_Geo":"#60a5fa","Hierarchical_Geo":"#f59e0b"}

        fig_r = go.Figure()
        for algo, scores in algo_scores.items():
            vals = [scores["Silhouette"]/max_sil if max_sil>0 else 0,
                    scores["DB_inv"], scores["Speed"],
                    scores["Scalability"], scores["Interpretability"],
                    scores["Silhouette"]/max_sil if max_sil>0 else 0]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]],
                name=algo, line=dict(color=colors_r.get(algo,"#ffffff"), width=2),
                fill="toself", opacity=0.15,
            ))
        fig_r.update_layout(
            polar=dict(bgcolor="#161b22",
                        radialaxis=dict(visible=True, range=[0,1], gridcolor="#374151"),
                        angularaxis=dict(gridcolor="#374151")),
            paper_bgcolor="#161b22", template="plotly_dark",
            height=420, showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.warning("Metric data required. Run `scripts/train.py`.")

# ── Tab 4: Best model ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🏆 Best Model Selection & Justification")
    if not sil_df.empty:
        geo_best = sil_df[sil_df["Algorithm"].str.contains("Geo",na=False)].copy()
        geo_best["Silhouette_n"] = pd.to_numeric(geo_best["Silhouette"], errors="coerce").fillna(0)
        geo_best["DB_n"] = pd.to_numeric(geo_best["Davies_Bouldin"], errors="coerce").fillna(9999)

        if not geo_best.empty:
            best_sil_row = geo_best.loc[geo_best["Silhouette_n"].idxmax()]
            best_db_row  = geo_best.loc[geo_best["DB_n"].idxmin()]

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown(f"""
                <div class="winner-box">
                    <h4 style='color:#4ade80;margin:0 0 0.5rem'>🥇 Best Silhouette Score</h4>
                    <h3 style='color:#e6edf3;margin:0'>{best_sil_row['Algorithm']}</h3>
                    <p style='color:#8b949e;font-size:0.85rem;margin:0.5rem 0'>
                        Silhouette = <b style='color:#4ade80'>{best_sil_row['Silhouette_n']:.4f}</b><br>
                        (Higher = better-defined, well-separated clusters)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col_b2:
                st.markdown(f"""
                <div class="winner-box">
                    <h4 style='color:#60a5fa;margin:0 0 0.5rem'>🎯 Best Davies-Bouldin</h4>
                    <h3 style='color:#e6edf3;margin:0'>{best_db_row['Algorithm']}</h3>
                    <p style='color:#8b949e;font-size:0.85rem;margin:0.5rem 0'>
                        DB Index = <b style='color:#60a5fa'>{best_db_row['DB_n']:.4f}</b><br>
                        (Lower = tighter clusters with less overlap)
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 📌 Recommendation for Deployment

    | Criterion | K-Means | DBSCAN | Hierarchical |
    |---|---|---|---|
    | Silhouette Score | Good | **Best** | Good |
    | Davies-Bouldin | Good | **Best** | Good |
    | Handles Noise | ❌ | ✅ | ❌ |
    | Scales to 500K | ✅ | ✅ | ❌ (slow) |
    | Interpretable | ✅ | ✅ | ✅ |
    | No k required | ❌ | ✅ | ❌ |

    **✅ Recommended: DBSCAN** for production deployment.
    DBSCAN naturally discovers crime-dense areas without requiring a pre-specified cluster count.
    It filters out noise (isolated crimes) and adapts to Chicago's organic neighborhood structure.
    Use **K-Means** when patrol zones need fixed, equal-sized patrol areas (budget-driven allocation).
    """)

# ── Tab 5: Experiment details ─────────────────────────────────────────────────
with tab5:
    st.markdown("### 📁 Experiment Run Details")
    st.info("All model runs were tracked under the `PatrolIQ_Crime_Analysis` MLflow experiment.")

    detail_rows = [
        {"Run": "KMeans_Geographic",    "Algorithm": "K-Means",       "Type": "Geo Clustering",   "Params Logged": "n_clusters=8, random_state=42",             "Metrics": "silhouette, davies_bouldin",       "Artifacts": "model pkl, metrics JSON"},
        {"Run": "DBSCAN_Geographic",    "Algorithm": "DBSCAN",        "Type": "Geo Clustering",   "Params Logged": "eps=0.15, min_samples=10",                   "Metrics": "silhouette, davies_bouldin, noise_pct", "Artifacts": "metrics JSON"},
        {"Run": "Hierarchical_Geographic","Algorithm": "Agglomerative","Type": "Geo Clustering",  "Params Logged": "n_clusters=8, linkage=ward",                 "Metrics": "silhouette, davies_bouldin",       "Artifacts": "model pkl, dendrogram PNG"},
        {"Run": "KMeans_Temporal",      "Algorithm": "K-Means",       "Type": "Temporal Clustering","Params Logged": "n_clusters=4, type=temporal",              "Metrics": "silhouette, davies_bouldin",       "Artifacts": "model pkl"},
        {"Run": "PCA_DimensionalityReduction","Algorithm": "PCA",     "Type": "Dim. Reduction",   "Params Logged": "n_components=2, n_for_70pct",               "Metrics": "variance_explained_2d, n_components_for_70pct", "Artifacts": "pca model, scree CSV, loadings CSV"},
        {"Run": "tSNE_Visualization",   "Algorithm": "t-SNE",         "Type": "Dim. Reduction",   "Params Logged": "perplexity=30, max_iter=300, n_components=2","Metrics": "kl_divergence",                   "Artifacts": "tsne result CSV"},
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    mlruns_path = os.path.join(os.path.dirname(__file__), "..", "mlruns")
    st.markdown(f"""
    **To launch the MLflow UI locally:**
    ```bash
    source venv/bin/activate
    mlflow ui --backend-store-uri {os.path.abspath(mlruns_path)}
    # Then open http://localhost:5000
    ```
    """)
