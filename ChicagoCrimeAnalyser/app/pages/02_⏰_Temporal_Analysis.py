"""
PatrolIQ - Temporal Analysis Page
Hourly/weekday/seasonal dashboards + temporal cluster visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Temporal Analysis — PatrolIQ", layout="wide", page_icon="⏰")
BASE_DIR = Path(__file__).resolve().parents[2]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
html,body,.stApp{background:#0a0e1a;}
.page-title{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#4facfe,#00f2fe);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem;}
.sub-text{color:#8892b0;margin-bottom:2rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#161b2e);border-right:1px solid #2a3050;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">⏰ Temporal Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Explore when crimes occur most frequently across hours, weekdays, and seasons</div>', unsafe_allow_html=True)

DATA_PATH = BASE_DIR / "data" / "processed" / "model_ready_data.csv"
REPORT_DIR = BASE_DIR / "reports" / "summaries"
FIG_DIR    = BASE_DIR / "reports" / "figures"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)

try:
    df = load_data()
    has_data = True
except:
    has_data = False
    st.error("Run `clean_data.py` and `feature_engineering.py` first.")

if has_data:
    # ── Filters ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        if "Season" in df.columns:
            seasons = ["All"] + sorted(df["Season"].dropna().unique().tolist())
            sel_season = st.selectbox("Season", seasons)
        if "Primary Type" in df.columns:
            top_crimes = df["Primary Type"].value_counts().head(15).index.tolist()
            sel_crime = st.selectbox("Crime Type", ["All"] + top_crimes)

        filtered = df.copy()
        if "Season" in df.columns and sel_season != "All":
            filtered = filtered[filtered["Season"] == sel_season]
        if "Primary Type" in df.columns and sel_crime != "All":
            filtered = filtered[filtered["Primary Type"] == sel_crime]
        st.metric("Records shown", f"{len(filtered):,}")

    # ── Tab layout ───────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🕐 Hourly", "📅 Weekday", "🌸 Seasonal", "🔮 Clusters"])

    with tab1:
        st.subheader("Crime Volume by Hour of Day")
        hourly = filtered["Hour"].value_counts().sort_index().reset_index()
        hourly.columns = ["Hour","Count"]
        fig = px.bar(hourly, x="Hour", y="Count", color="Count",
                     color_continuous_scale="Inferno",
                     labels={"Count":"Crimes","Hour":"Hour of Day"})
        fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
        st.plotly_chart(fig, use_container_width=True)

        # Line chart for trend
        fig2 = px.line(hourly, x="Hour", y="Count", markers=True,
                       title="Hourly Crime Trend", color_discrete_sequence=["#4facfe"])
        fig2.add_vrect(x0=22, x1=24, fillcolor="#f093fb", opacity=0.1, annotation_text="High Risk")
        fig2.add_vrect(x0=0, x1=3, fillcolor="#f093fb", opacity=0.1)
        fig2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        # Weekday × Hour heatmap
        wdh_path = BASE_DIR / "reports" / "figures" / "weekday_hour_heatmap.png"
        if wdh_path.exists():
            st.image(str(wdh_path), caption="Weekday × Hour Crime Intensity Heatmap", use_container_width=True)

        if "Weekday" in filtered.columns and "Hour" in filtered.columns:
            cross = filtered.groupby(["Weekday","Hour"]).size().reset_index(name="Count")
            cross["Day_Name"] = cross["Weekday"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
            fig_cross = px.density_heatmap(cross, x="Hour", y="Day_Name", z="Count",
                                           color_continuous_scale="Magma",
                                           title="Crime Intensity: Weekday × Hour")
            fig_cross.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
            st.plotly_chart(fig_cross, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if "Season" in filtered.columns:
                season_data = filtered["Season"].value_counts().reset_index()
                season_data.columns = ["Season","Count"]
                fig_s = px.pie(season_data, values="Count", names="Season", hole=0.45,
                               title="Crimes by Season",
                               color_discrete_sequence=["#667eea","#764ba2","#f093fb","#4facfe"])
                fig_s.update_layout(paper_bgcolor="#0a0e1a", font_color="#ccd6f6")
                st.plotly_chart(fig_s, use_container_width=True)
        with col2:
            if "Month" in filtered.columns:
                monthly = filtered["Month"].value_counts().sort_index().reset_index()
                monthly.columns = ["Month","Count"]
                month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                monthly["Month_Name"] = monthly["Month"].map(month_names)
                fig_m = px.bar(monthly, x="Month_Name", y="Count", title="Crimes by Month",
                               color="Count", color_continuous_scale="thermal")
                fig_m.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
                st.plotly_chart(fig_m, use_container_width=True)

        # Weekend vs weekday
        if "Is_Weekend" in filtered.columns:
            wd_data = filtered["Is_Weekend"].map({0:"Weekday",1:"Weekend"}).value_counts().reset_index()
            wd_data.columns = ["Type","Count"]
            fig_wd = px.bar(wd_data, x="Type", y="Count", color="Type",
                            title="Weekday vs Weekend Crime Volume",
                            color_discrete_map={"Weekday":"#667eea","Weekend":"#f093fb"})
            fig_wd.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6", showlegend=False)
            st.plotly_chart(fig_wd, use_container_width=True)

    with tab4:
        if all(c in filtered.columns for c in ["Hour_Sin","Hour_Cos"]):
            sample = filtered.sample(min(8_000, len(filtered)), random_state=42)
            color_col = "Time_Label" if "Time_Label" in sample.columns else "Hour"
            fig_t = px.scatter(sample, x="Hour_Cos", y="Hour_Sin",
                               color=color_col, hover_data=["Hour"] if "Hour" in sample.columns else None,
                               title="Temporal Cluster Distribution (Cyclic Hour Encoding)",
                               color_continuous_scale="Viridis", opacity=0.7)
            fig_t.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
            st.plotly_chart(fig_t, use_container_width=True)
            st.caption("Each point represents a crime record. Angle = time of day. Distance from center = pattern intensity.")
