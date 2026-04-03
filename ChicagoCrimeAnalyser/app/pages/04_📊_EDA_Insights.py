"""
PatrolIQ - EDA Insights Page
Crime type rankings, arrest rates, dataset stats from summary CSVs
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EDA Insights — PatrolIQ", layout="wide", page_icon="📊")
BASE_DIR = Path(__file__).resolve().parents[2]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
html,body,.stApp{background:#0a0e1a;}
.page-title{font-family:'Outfit',sans-serif;font-size:2.8rem;font-weight:900;background:linear-gradient(135deg,#43e97b,#38f9d7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:.5rem;}
.sub-text{color:#8892b0;margin-bottom:2rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1117,#161b2e);border-right:1px solid #2a3050;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 EDA Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Comprehensive exploratory analysis of Chicago crime patterns</div>', unsafe_allow_html=True)

SUM_DIR = BASE_DIR / "reports" / "summaries"
FIG_DIR = BASE_DIR / "reports" / "figures"

def dark_layout(fig):
    fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#161b2e", font_color="#ccd6f6")
    return fig

tab1, tab2, tab3, tab4 = st.tabs(["🔢 Crime Types", "🚔 Arrest Rates", "🗺️ Static EDA Maps", "📋 Data Quality"])

with tab1:
    crime_path = SUM_DIR / "crime_counts.csv"
    if crime_path.exists():
        crime_df = pd.read_csv(crime_path)
        top_n = st.slider("Show top N crime types", 5, 33, 20)
        display = crime_df.head(top_n).sort_values("Count")
        fig = px.bar(display, x="Count", y="Primary Type", orientation="h",
                     color="Count", color_continuous_scale="Viridis",
                     title=f"Top {top_n} Crime Types in Chicago")
        dark_layout(fig)
        fig.update_layout(height=max(400, top_n * 28))
        st.plotly_chart(fig, use_container_width=True)

        # Donut chart
        col1, col2 = st.columns([1,1])
        with col1:
            top5 = crime_df.head(5)
            other = pd.DataFrame([{"Primary Type": "Others", "Count": crime_df.iloc[5:]["Count"].sum()}])
            pie_df = pd.concat([top5, other])
            fig_pie = px.pie(pie_df, values="Count", names="Primary Type", hole=0.4,
                             title="Crime Share — Top 5 vs Others")
            dark_layout(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.markdown("#### 🔢 Top 10 Crime Counts")
            st.dataframe(crime_df.head(10).reset_index(drop=True), use_container_width=True, height=300)
    else:
        st.warning("Run `eda_pipeline.py` to generate crime count data.")

with tab2:
    arrest_path = SUM_DIR / "arrest_domestic_by_type.csv"
    if arrest_path.exists():
        arrest_df = pd.read_csv(arrest_path)
        col1, col2 = st.columns(2)
        with col1:
            fig_ar = px.bar(arrest_df.sort_values("arrest_rate", ascending=True).head(20),
                            x="arrest_rate", y="Primary Type", orientation="h",
                            color="arrest_rate", color_continuous_scale="RdYlGn",
                            title="Arrest Rate by Crime Type (Top 20)")
            dark_layout(fig_ar)
            st.plotly_chart(fig_ar, use_container_width=True)
        with col2:
            fig_dom = px.bar(arrest_df.sort_values("domestic_rate", ascending=True).head(20),
                             x="domestic_rate", y="Primary Type", orientation="h",
                             color="domestic_rate", color_continuous_scale="Oranges",
                             title="Domestic Incident Rate by Crime Type")
            dark_layout(fig_dom)
            st.plotly_chart(fig_dom, use_container_width=True)

        # Summary stats
        avg_arrest = arrest_df["arrest_rate"].mean()
        avg_domestic = arrest_df["domestic_rate"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Arrest Rate", f"{avg_arrest:.1%}")
        c2.metric("Avg Domestic Rate", f"{avg_domestic:.1%}")
        c3.metric("Crime Types Analyzed", len(arrest_df))
    else:
        st.warning("Run `eda_pipeline.py` to generate arrest rate data.")

with tab3:
    # Static EDA images
    images = {
        "Top 20 Crime Types": FIG_DIR / "top20_crime_types.png",
        "Crimes by Hour": FIG_DIR / "crimes_by_hour.png",
        "Weekday × Hour Heatmap": FIG_DIR / "weekday_hour_heatmap.png",
        "Crimes by Season": FIG_DIR / "crimes_by_season.png",
        "Arrest Rate by Type": FIG_DIR / "arrest_rate_by_type.png",
        "Geographic Scatter (60k)": FIG_DIR / "geo_scatter.png",
    }
    existing = {k: v for k, v in images.items() if v.exists()}
    if existing:
        cols = st.columns(2)
        for i, (title, path) in enumerate(existing.items()):
            with cols[i % 2]:
                st.image(str(path), caption=title, use_container_width=True)
    else:
        st.warning("Run `eda_pipeline.py` to generate EDA charts.")

with tab4:
    import json
    val_path = SUM_DIR / "data_validation_summary.json"
    if val_path.exists():
        with open(val_path) as f: val = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{val.get('records', 'N/A'):,}" if isinstance(val.get('records'), int) else "N/A")
        c2.metric("Columns", val.get('columns', 'N/A'))
        c3.metric("Duplicates", val.get('duplicates', 'N/A'))
        c4.metric("Geo Outliers", val.get('geo_outliers', 'N/A'))

        if "date_range" in val:
            st.info(f"📅 Date Range: {val['date_range'][0]} → {val['date_range'][1]}")
        if "missing_values" in val and val["missing_values"]:
            st.warning("🧩 Columns with missing values detected (post-processing):")
            st.json(val["missing_values"])
        else:
            st.success("✅ No missing values in the processed dataset")

        txt_path = SUM_DIR / "data_validation_report.txt"
        if txt_path.exists():
            with st.expander("📄 Full Validation Report"):
                st.text(txt_path.read_text())
    else:
        st.warning("Run `validate_data.py` to generate the data quality report.")
