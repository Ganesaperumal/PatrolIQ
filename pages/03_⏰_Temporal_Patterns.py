"""
PatrolIQ Page 3 — Temporal Patterns
Hour×Day heatmap, monthly trends, weekday vs weekend, seasonal, violent peaks.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os

st.set_page_config(page_title="Temporal Patterns | PatrolIQ", page_icon="⏰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.page-header { background:linear-gradient(135deg,#0d1117,#0e1a2e); border:1px solid #21262d; border-radius:16px; padding:1.8rem 2rem; margin-bottom:1.5rem; }
.page-title  { font-size:2rem; font-weight:900; color:#e6edf3; margin:0; }
.page-sub    { color:#8b949e; font-size:0.9rem; margin-top:0.3rem; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1117,#0e1a2e); border-right:1px solid #21262d; }
.sidebar-brand { text-align: center; padding: 1.5rem 0 1rem; }
.sidebar-logo { font-size: 2.5rem; }
.sidebar-name { font-size: 1.2rem; font-weight: 800; color: #4ade80; }
.sidebar-tag  { font-size: 0.7rem; color: #8b949e; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

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

CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")

@st.cache_data(ttl=3600, show_spinner="Loading temporal data …")
def load_data():
    p = os.path.join(CLEAN_DIR, "cleaned_crimes.parquet")
    return pd.read_parquet(p,
                           columns=["Primary Type","Hour","Day_of_Week","Day_Num",
                                    "Month","Year","Season","Is_Weekend","Is_Night",
                                    "Arrest","Crime_Severity_Score"])

@st.cache_data(ttl=3600)
def load_temporal_clusters():
    p = os.path.join(CLEAN_DIR, "temporal_sample.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

df      = load_data()
temp_cl = load_temporal_clusters()

st.markdown("""
<div class="page-header">
    <div class="page-title">⏰ Temporal Crime Patterns</div>
    <div class="page-sub">When do crimes happen? Uncover hourly, daily, monthly, and seasonal rhythms</div>
</div>
""", unsafe_allow_html=True)

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🔽 Filter Options"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        years = sorted(df["Year"].unique())
        sel_year = st.select_slider("Year Range", options=years, value=(years[0],years[-1]))
    with col_f2:
        crime_types = ["All"] + sorted(df["Primary Type"].unique())
        sel_crime = st.selectbox("Crime Type", crime_types)

filtered = df[(df["Year"] >= sel_year[0]) & (df["Year"] <= sel_year[1])]
if sel_crime != "All":
    filtered = filtered[filtered["Primary Type"] == sel_crime]

# ── Row 1: Hour × Day Heatmap ─────────────────────────────────────────────────
st.markdown("#### 🕐 Crime Intensity Heatmap — Hour × Day of Week")
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
pivot = (filtered.groupby(["Day_of_Week","Hour"]).size()
                .unstack(fill_value=0)
                .reindex(day_order))
fig1 = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[f"{h:02d}:00" for h in pivot.columns],
    y=pivot.index.tolist(),
    colorscale="YlOrRd",
    hoverongaps=False,
    hovertemplate="<b>%{y}</b> at <b>%{x}</b><br>Crimes: %{z}<extra></extra>",
))
fig1.update_layout(
    template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    margin=dict(l=10,r=10,t=20,b=10), height=360,
    xaxis=dict(title="Hour of Day", tickangle=-45),
    yaxis=dict(title=""),
)
st.plotly_chart(fig1, use_container_width=True)
st.caption("🔴 Dark red = peak crime hours | Yellow = moderate | White = low activity")
st.divider()

# ── Row 2: Hourly + Monthly ────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### ⏱ Hourly Crime Volume")
    hourly = filtered.groupby("Hour").size().reset_index(name="Crimes")
    hourly["Label"] = hourly["Hour"].apply(lambda h:
        "🌙 Night" if h < 6 else ("☀️ Morning" if h < 12 else ("🌤 Afternoon" if h < 18 else "🌆 Evening")))
    color_map = {"🌙 Night":"#6366f1","☀️ Morning":"#f59e0b","🌤 Afternoon":"#4ade80","🌆 Evening":"#f87171"}
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=hourly["Hour"], y=hourly["Crimes"],
                           marker=dict(color=[color_map[l] for l in hourly["Label"]], opacity=0.85),
                           hovertemplate="Hour %{x}:00 → %{y:,} crimes<extra></extra>"))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        margin=dict(l=10,r=10,t=20,b=10), height=300,
                        xaxis=dict(title="Hour (24h)"), yaxis=dict(title="Crime Count"))
    st.plotly_chart(fig2, use_container_width=True)
    peak_h = int(hourly.loc[hourly["Crimes"].idxmax(),"Hour"])
    st.markdown(f"**⚡ Peak Hour:** `{peak_h}:00` ({peak_h if peak_h>=12 else 12}:{00 if peak_h%12==0 else ''} {'PM' if peak_h>=12 else 'AM'})")

with col_b:
    st.markdown("#### 📅 Monthly Crime Volume")
    mon_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = filtered.groupby("Month").size().reset_index(name="Crimes")
    monthly["Month Name"] = monthly["Month"].map(mon_map)
    fig3 = px.line(monthly, x="Month Name", y="Crimes",
                    markers=True, template="plotly_dark",
                    color_discrete_sequence=["#4ade80"])
    fig3.add_bar(x=monthly["Month Name"], y=monthly["Crimes"],
                  marker_color="#4ade80", opacity=0.2, name="Volume")
    fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        showlegend=False, margin=dict(l=10,r=10,t=20,b=10), height=300,
                        xaxis=dict(categoryorder="array",categoryarray=list(mon_map.values())))
    st.plotly_chart(fig3, use_container_width=True)
    peak_m = int(filtered["Month"].value_counts().index[0])
    st.markdown(f"**📈 Peak Month:** `{mon_map[peak_m]}` — highest crime volume")

st.divider()

# ── Row 3: Weekday vs Weekend + Season ────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 📊 Weekday vs Weekend Crime Patterns")
    filtered["WE_Label"] = filtered["Is_Weekend"].map({1:"Weekend","Weekend":"Weekend",
                                                        0:"Weekday","Weekday":"Weekday"})
    try:
        wday = filtered.groupby(["Day_of_Week","Is_Weekend"]).size().reset_index(name="Crimes")
        wday["Type"] = wday["Is_Weekend"].apply(lambda x: "Weekend" if x==1 else "Weekday")
    except Exception:
        wday = filtered.groupby("Day_of_Week").size().reset_index(name="Crimes")
        wday["Type"] = wday["Day_of_Week"].apply(
            lambda d: "Weekend" if d in ["Saturday","Sunday"] else "Weekday")
    fig4 = px.bar(wday, x="Day_of_Week", y="Crimes", color="Type",
                   color_discrete_map={"Weekend":"#f59e0b","Weekday":"#4ade80"},
                   template="plotly_dark",
                   category_orders={"Day_of_Week":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]})
    fig4.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        margin=dict(l=10,r=10,t=20,b=10), height=300,
                        legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.markdown("#### 🌡️ Crime Rate by Season")
    season = filtered.groupby("Season").agg(
        Crimes=("Season","count"),
        Arrest_Rate=("Arrest","mean"),
    ).reset_index()
    season["Arrest Rate %"] = (season["Arrest_Rate"]*100).round(1)
    season_order = ["Spring","Summer","Fall","Winter"]
    season_colors = {"Summer":"#f59e0b","Fall":"#ef4444","Winter":"#60a5fa","Spring":"#4ade80"}
    fig5 = px.bar(season, x="Season", y="Crimes",
                   color="Season", color_discrete_map=season_colors,
                   template="plotly_dark",
                   category_orders={"Season":season_order})
    fig5.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        showlegend=False, margin=dict(l=10,r=10,t=20,b=10), height=300)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ── Row 4: Violent crimes peak + Temporal cluster results ─────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.markdown("#### 🔴 Violent Crimes Peak Hours")
    violent_types = ["ASSAULT","BATTERY","HOMICIDE","ROBBERY","KIDNAPPING",
                      "CRIMINAL SEXUAL ASSAULT","SEX OFFENSE"]
    violent = filtered[filtered["Primary Type"].isin(violent_types)]
    if len(violent) > 0:
        v_hourly = violent.groupby("Hour").size().reset_index(name="Violent Crimes")
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=v_hourly["Hour"], y=v_hourly["Violent Crimes"],
                                   mode="lines+markers", line=dict(color="#ef4444",width=2.5),
                                   fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
                                   hovertemplate="Hour %{x}:00 → %{y:,} violent crimes<extra></extra>"))
        fig6.update_layout(template="plotly_dark", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                            margin=dict(l=10,r=10,t=20,b=10), height=300,
                            xaxis=dict(title="Hour (24h)"), yaxis=dict(title="Violent Crimes"))
        st.plotly_chart(fig6, use_container_width=True)
        peak_v = int(v_hourly.loc[v_hourly["Violent Crimes"].idxmax(),"Hour"])
        st.markdown(f"**🚨 Peak Violent Crime Hour:** `{peak_v}:00` — maximum patrol recommended")
    else:
        st.info("Apply a broader filter to see violent crime patterns.")

with col_f:
    st.markdown("#### 🔵 Temporal Cluster Profiles (K-Means k=4)")
    if not temp_cl.empty and "Temporal_Cluster" in temp_cl.columns:
        cl_stats = temp_cl.groupby("Temporal_Cluster").agg(
            Avg_Hour=("Hour","mean"),
            Avg_Month=("Month","mean"),
            Crime_Severity=("Crime_Severity_Score","mean"),
            Count=("Temporal_Cluster","count"),
        ).reset_index().round(2)
        cl_stats.rename(columns={"Temporal_Cluster":"Cluster"}, inplace=True)

        cluster_names = {0:"Night Crime Pattern",1:"Rush Hour Crime",
                          2:"Weekend Activity",3:"Afternoon Crime Wave"}
        cl_stats["Profile"] = cl_stats["Cluster"].map(cluster_names)

        fig7 = px.scatter(cl_stats, x="Avg_Hour", y="Crime_Severity",
                           size="Count", color="Profile",
                           color_discrete_sequence=["#4ade80","#60a5fa","#f59e0b","#f87171"],
                           template="plotly_dark",
                           labels={"Avg_Hour":"Avg Hour of Day","Crime_Severity":"Avg Severity Score"})
        fig7.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                            margin=dict(l=10,r=10,t=20,b=10), height=300)
        st.plotly_chart(fig7, use_container_width=True)
        st.dataframe(cl_stats[["Cluster","Profile","Avg_Hour","Crime_Severity","Count"]],
                      use_container_width=True, hide_index=True)
    else:
        st.info("Run `scripts/train.py` to generate temporal cluster data.")

# ── Insights ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💡 Temporal Insights for Patrol Planning")
c1, c2, c3 = st.columns(3)
hourly_agg = filtered.groupby("Hour").size()
peak_hr = int(hourly_agg.idxmax())
with c1:
    st.success(f"""
**🌙 Night Patrol Priority**  
Peak crime hour: **{peak_hr}:00**  
Late night (10PM–2AM) sees the most violent crimes.  
→ Double patrol in high-risk zones during these hours.
    """)
with c2:
    summer_crimes = filtered[filtered["Season"]=="Summer"].shape[0]
    winter_crimes = filtered[filtered["Season"]=="Winter"].shape[0]
    summer_pct = summer_crimes / len(filtered) * 100 if len(filtered) > 0 else 0
    st.warning(f"""
**☀️ Seasonal Resource Allocation**  
Summer = {summer_pct:.0f}% of annual crimes.  
Winter sees significantly fewer outdoor crimes.  
→ Increase summer street presence by 30–40%.
    """)
with c3:
    we_crimes = filtered[filtered["Is_Weekend"]==1].shape[0]
    we_pct = we_crimes / len(filtered) * 100 if len(filtered) > 0 else 0
    st.error(f"""
**📅 Weekend Surge**  
{we_pct:.0f}% of crimes occur on weekends (Sat–Sun).  
Friday nights are the transition peak.  
→ Deploy specialized weekend patrol units.
    """)
