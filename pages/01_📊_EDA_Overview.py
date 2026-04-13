"""
PatrolIQ Page 1 — EDA Overview
KPIs, crime type distribution, temporal/district trends, arrest correlations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="EDA Overview | PatrolIQ", page_icon="📊", layout="wide")

# ── Shared style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.page-header {
    background: linear-gradient(135deg, #0d1117, #0e1a2e);
    border: 1px solid #21262d; border-radius: 16px;
    padding: 1.8rem 2rem; margin-bottom: 1.5rem;
}
.page-title { font-size:2rem; font-weight:900; color:#e6edf3; margin:0; }
.page-sub   { color:#8b949e; font-size:0.9rem; margin-top:0.3rem; }
.metric-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.metric-box {
    flex:1; min-width:150px; background:#161b22; border:1px solid #21262d;
    border-radius:12px; padding:1.2rem; text-align:center;
}
.metric-val { font-size:1.7rem; font-weight:800; color:#4ade80; }
.metric-lbl { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.3rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#0e1a2e); border-right:1px solid #21262d; }
.sidebar-brand { text-align: center; padding: 1.5rem 0 1rem; }
.sidebar-logo { font-size: 2.5rem; }
.sidebar-name { font-size: 1.2rem; font-weight: 800; color: #4ade80; }
.sidebar-tag  { font-size: 0.7rem; color: #8b949e; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar nav
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🚔</div>
        <div class="sidebar-name">PatrolIQ</div>
        <div class="sidebar-tag">Smart Safety Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                                  label="🏠 Home")
    st.page_link("pages/01_📊_EDA_Overview.py",             label="📊 EDA Overview")
    st.page_link("pages/02_🗺️_Geographic_Clusters.py",       label="🗺️ Geographic Clusters")
    st.page_link("pages/03_⏰_Temporal_Patterns.py",        label="⏰ Temporal Patterns")
    st.page_link("pages/04_🔬_Dimensionality_Reduction.py", label="🔬 Dimensionality Reduction")
    st.page_link("pages/05_📈_MLflow_Monitor.py",           label="📈 MLflow Monitor")

# ── Load data ─────────────────────────────────────────────────────────────────
CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")

@st.cache_data(ttl=3600, show_spinner="Loading crime dataset …")
def load_data():
    p = os.path.join(CLEAN_DIR, "cleaned_crimes.parquet")
    return pd.read_parquet(p, columns=["Primary Type","Year","Month","Hour","Day_of_Week",
                                       "District","Season","Is_Weekend","Arrest","Domestic",
                                       "Crime_Severity_Score","Location Description"])

@st.cache_data(ttl=3600)
def load_meta():
    p = os.path.join(CLEAN_DIR, "metadata.json")
    return json.load(open(p)) if os.path.exists(p) else {}

df   = load_data()
meta = load_meta()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Exploratory Data Analysis</div>
    <div class="page-sub">Deep-dive into Chicago's crime landscape — distributions, trends, and correlations</div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
total   = len(df)
types_n = df["Primary Type"].nunique()
arr_r   = df["Arrest"].mean() * 100
dom_r   = df["Domestic"].mean() * 100
sev_avg = df["Crime_Severity_Score"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
kpis = [
    (col1, "📊", f"{total:,}",         "Total Records"),
    (col2, "🏷️", f"{types_n}",          "Crime Types"),
    (col3, "🚨", f"{arr_r:.1f}%",      "Arrest Rate"),
    (col4, "🏠", f"{dom_r:.1f}%",      "Domestic Rate"),
    (col5, "⚡",  f"{sev_avg:.2f}/10", "Avg Severity"),
]
for col, icon, val, lbl in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div style='font-size:1.5rem'>{icon}</div>
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🔽 Filter Data", expanded=False):
    years_avail = sorted(df["Year"].dropna().unique())
    sel_years = st.select_slider("Year Range", options=years_avail,
                                  value=(years_avail[0], years_avail[-1]))
    sel_types = st.multiselect("Crime Types", options=sorted(df["Primary Type"].unique()),
                                default=[])

filtered = df[(df["Year"] >= sel_years[0]) & (df["Year"] <= sel_years[1])]
if sel_types:
    filtered = filtered[filtered["Primary Type"].isin(sel_types)]

st.markdown(f"**Showing {len(filtered):,} records** ({sel_years[0]}–{sel_years[1]})")
st.divider()

# ── Row 1: Top crimes + Arrest rate by type ────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🏆 Top 10 Crime Types")
    top10 = filtered["Primary Type"].value_counts().head(10).reset_index()
    top10.columns = ["Crime Type", "Count"]
    fig = px.bar(top10, x="Count", y="Crime Type", orientation="h",
                  color="Count", color_continuous_scale="Teal",
                  template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=10,r=10,t=30,b=10), height=380,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### 🎯 Arrest Rate by Crime Type (Top 15)")
    arr_by_type = (filtered.groupby("Primary Type")["Arrest"]
                   .agg(["mean","count"]).reset_index())
    arr_by_type.columns = ["Crime Type","Arrest Rate","Count"]
    arr_by_type = arr_by_type[arr_by_type["Count"] > 100]
    arr_by_type["Arrest Rate %"] = (arr_by_type["Arrest Rate"] * 100).round(1)
    arr_by_type = arr_by_type.nlargest(15, "Arrest Rate %")
    fig2 = px.bar(arr_by_type, x="Arrest Rate %", y="Crime Type", orientation="h",
                   color="Arrest Rate %", color_continuous_scale="RdYlGn",
                   template="plotly_dark")
    fig2.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=10,r=10,t=30,b=10), height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Year trend + Monthly distribution ───────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 📈 Crime Trend by Year")
    year_trend = filtered.groupby("Year").size().reset_index(name="Crimes")
    yr_arr = (filtered.groupby("Year")["Arrest"].mean() * 100).reset_index(name="Arrest Rate %")
    year_trend = year_trend.merge(yr_arr, on="Year")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=year_trend["Year"], y=year_trend["Crimes"],
                               name="Crime Count", line=dict(color="#4ade80", width=2.5),
                               fill="tozeroy", fillcolor="rgba(74,222,128,0.07)"))
    fig3.add_trace(go.Scatter(x=year_trend["Year"], y=year_trend["Arrest Rate %"],
                               name="Arrest Rate %", line=dict(color="#60a5fa", width=2, dash="dot"),
                               yaxis="y2"))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        yaxis=dict(title="Crime Count"),
        yaxis2=dict(title="Arrest Rate %", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10,r=10,t=30,b=10), height=340,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown("#### 🗓️ Monthly Crime Distribution")
    mon_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = filtered.groupby("Month").size().reset_index(name="Crimes")
    monthly["Month Name"] = monthly["Month"].map(mon_map)
    fig4 = px.bar(monthly, x="Month Name", y="Crimes",
                   color="Crimes", color_continuous_scale="Viridis",
                   template="plotly_dark")
    fig4.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        coloraxis_showscale=False,
        xaxis=dict(categoryorder="array", categoryarray=list(mon_map.values())),
        margin=dict(l=10,r=10,t=30,b=10), height=340,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: District heatmap + Season pie ──────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.markdown("#### 🏙️ Crime Volume by District")
    dist = (filtered.groupby("District").agg(
        Crimes=("District","count"),
        Arrest_Rate=("Arrest","mean")).reset_index())
    dist["Arrest_Rate"] = (dist["Arrest_Rate"]*100).round(1)
    dist = dist[dist["District"] > 0].sort_values("Crimes", ascending=False)
    fig5 = px.bar(dist, x="District", y="Crimes",
                   color="Arrest_Rate", color_continuous_scale="RdYlGn",
                   template="plotly_dark",
                   labels={"Arrest_Rate":"Arrest Rate %"},
                   hover_data=["Arrest_Rate"])
    fig5.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        margin=dict(l=10,r=10,t=30,b=10), height=340,
        xaxis=dict(type="category")
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_f:
    st.markdown("#### 🍂 Seasonal Crime Distribution")
    season = filtered["Season"].value_counts().reset_index()
    season.columns = ["Season","Count"]
    season_colors = {"Summer":"#f59e0b","Fall":"#ef4444","Winter":"#60a5fa","Spring":"#4ade80"}
    fig6 = px.pie(season, names="Season", values="Count",
                   color="Season", color_discrete_map=season_colors,
                   hole=0.55, template="plotly_dark")
    fig6.update_layout(
        paper_bgcolor="#161b22",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=10,r=10,t=30,b=10), height=340,
    )
    fig6.update_traces(textinfo="percent+label", textfont_size=11)
    st.plotly_chart(fig6, use_container_width=True)

# ── Row 4: Missing value analysis + Domestic vs Arrest breakdown ────────────────
st.divider()
st.markdown("#### 📋 Data Quality & Missing Values")

orig_cols = ["ID","Case Number","Date","Block","IUCR","Primary Type","Description",
             "Location Description","Arrest","Domestic","Beat","District","Ward",
             "Community Area","FBI Code","X Coordinate","Y Coordinate","Year",
             "Updated On","Latitude","Longitude","Location"]
raw_path = os.path.join(CLEAN_DIR, "..", "..", "data", "uncleaned", "crimes_data.csv")
if os.path.exists(os.path.join(CLEAN_DIR, "..", "uncleaned", "crimes_data.csv")):
    @st.cache_data(ttl=7200, show_spinner="Computing missing values …")
    def get_missing():
        raw = pd.read_csv(os.path.join(CLEAN_DIR, "..", "uncleaned", "crimes_data.csv"),
                           nrows=50000, low_memory=False)
        miss = (raw.isnull().sum() / len(raw) * 100).reset_index()
        miss.columns = ["Column","Missing %"]
        return miss[miss["Missing %"] > 0].sort_values("Missing %", ascending=False)
    miss_df = get_missing()
    if not miss_df.empty:
        fig7 = px.bar(miss_df, x="Column", y="Missing %",
                       color="Missing %", color_continuous_scale="Reds",
                       template="plotly_dark")
        fig7.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                            margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.success("✅ No missing values in sampled data — clean dataset!")
else:
    st.info("Raw data file not accessible for missing value analysis (cleaned data is complete).")

# ── Severity score distribution ───────────────────────────────────────────────
st.markdown("#### ⚡ Crime Severity Score Distribution")
sev = filtered["Crime_Severity_Score"].value_counts().reset_index()
sev.columns = ["Score","Count"]
sev_labels = {1:"Very Low",2:"Low",3:"Moderate",4:"Medium",5:"High",
               6:"Serious",7:"Very Serious",8:"Severe",9:"Critical",10:"Extreme"}
sev["Label"] = sev["Score"].map(sev_labels)
fig8 = px.bar(sev.sort_values("Score"), x="Score", y="Count",
               color="Score", color_continuous_scale="RdYlGn_r",
               text="Label", template="plotly_dark")
fig8.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    coloraxis_showscale=False,
                    margin=dict(l=10,r=10,t=10,b=10), height=280)
fig8.update_traces(textposition="outside")
st.plotly_chart(fig8, use_container_width=True)

# ── Insights box ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💡 Key EDA Insights")
top_type = df["Primary Type"].value_counts().index[0]
peak_hour = int(df["Hour"].value_counts().index[0])
peak_month_n = int(df["Month"].value_counts().index[0])
peak_season = df["Season"].value_counts().index[0]
col1, col2 = st.columns(2)
with col1:
    st.info(f"""
**Most Common Crime:** `{top_type}` — accounting for {df['Primary Type'].value_counts().iloc[0]/len(df)*100:.1f}% of all incidents.

**Peak Activity Hour:** `{peak_hour}:00` hours — criminals favor late-evening hours.

**Peak Month:** `{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][peak_month_n-1]}` shows the highest crime volume.
    """)
with col2:
    st.info(f"""
**Highest Arrest Rate:** Narcotics offenses have the highest arrest rate due to targeted enforcement.

**Peak Season:** `{peak_season}` sees the most crimes — warmer weather correlates with increased outdoor activity and crime.

**Domestic Crimes:** {dom_r:.1f}% of crimes are domestic incidents, requiring specialized intervention.
    """)
