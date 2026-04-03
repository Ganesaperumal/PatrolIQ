"""
PatrolIQ - MLflow Monitoring Page
Live reads mlruns/mlflow.db and displays experiment runs, parameters, and metrics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from pathlib import Path

st.set_page_config(page_title="MLflow Monitoring — PatrolIQ", layout="wide", page_icon="📈")
BASE_DIR = Path(__file__).resolve().parents[2]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
html,body,.stApp{background:#0a0e1a;}
.page-title{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem;}
.sub-text{color:#8892b0;margin-bottom:2rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#161b2e);border-right:1px solid #2a3050;}
.run-card{background:#161b2e;border:1px solid #2a3050;border-radius:12px;padding:1.2rem;margin:.5rem 0;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📈 MLflow Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Track, compare and monitor all experiment runs across clustering and dimensionality reduction models</div>', unsafe_allow_html=True)

# Set MLflow tracking URI
mlflow_db = BASE_DIR / "mlflow.db"
if mlflow_db.exists():
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
else:
    st.warning("⚠️ MLflow database not found. Run the model scripts first.")
    st.stop()

@st.cache_data(ttl=30)
def get_all_runs():
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    all_rows = []
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
        for run in runs:
            row = {
                "Experiment": exp.name,
                "Run Name": run.info.run_name or run.info.run_id[:8],
                "Status": run.info.status,
                "Start": pd.to_datetime(run.info.start_time, unit="ms"),
            }
            row.update({f"param_{k}": v for k, v in run.data.params.items()})
            row.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            all_rows.append(row)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs found. Run the model scripts to populate MLflow experiments.")
else:
    # Experiment filter
    experiments = runs_df["Experiment"].unique().tolist()
    sel_exp = st.multiselect("Filter by Experiment", experiments, default=experiments)
    display_df = runs_df[runs_df["Experiment"].isin(sel_exp)] if sel_exp else runs_df

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", len(display_df))
    col2.metric("Experiments", display_df["Experiment"].nunique())
    col3.metric("Completed Runs", len(display_df[display_df["Status"] == "FINISHED"]))

    st.markdown("---")

    # Silhouette comparison chart
    sil_col = "metric_silhouette_score"
    db_col  = "metric_davies_bouldin_score"
    if sil_col in display_df.columns:
        st.subheader("📊 Silhouette Score Comparison")
        chart_df = display_df[["Run Name","Experiment", sil_col]].dropna()
        fig_sil = px.bar(chart_df, x="Run Name", y=sil_col, color="Experiment",
                         barmode="group", title="Silhouette Score per Run",
                         color_discrete_sequence=["#667eea","#f093fb","#4facfe"])
        fig_sil.add_hline(y=0.5, line_dash="dash", line_color="#43e97b", annotation_text="Target 0.5")
        fig_sil.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
        st.plotly_chart(fig_sil, use_container_width=True)

    if db_col in display_df.columns:
        st.subheader("📊 Davies-Bouldin Index Comparison")
        chart_db = display_df[["Run Name","Experiment", db_col]].dropna()
        fig_db = px.bar(chart_db, x="Run Name", y=db_col, color="Experiment",
                        barmode="group", title="Davies-Bouldin Index (Lower = Better)",
                        color_discrete_sequence=["#667eea","#f093fb","#4facfe"])
        fig_db.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
        st.plotly_chart(fig_db, use_container_width=True)

    # Full runs table
    st.subheader("📋 All Experiment Runs")
    disp_cols = ["Experiment","Run Name","Status","Start"] + \
                [c for c in display_df.columns if c.startswith("metric_") or c.startswith("param_")]
    disp_cols = [c for c in disp_cols if c in display_df.columns]
    st.dataframe(display_df[disp_cols].reset_index(drop=True), use_container_width=True)

    # MLflow UI link
    st.info("💡 **Pro Tip**: Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` in your terminal to access the full MLflow web interface.")
