"""
PatrolIQ - Exploratory Data Analysis Pipeline
Best-of-both: Folium heatmap + weekday×hour cross-heatmap + Plotly interactives + CSV exports
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "sample_500000_rows.csv"
FIG_DIR   = BASE_DIR / "reports" / "figures"
SUM_DIR   = BASE_DIR / "reports" / "summaries"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR.mkdir(parents=True, exist_ok=True)

print("📂 Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
print(f"✅ Loaded: {len(df):,} records")

# ── 1. Crime Type Distribution ────────────────────────────
print("📊 Crime type distribution...")
crime_counts = df["Primary Type"].value_counts()
crime_df = crime_counts.reset_index()
crime_df.columns = ["Primary Type", "Count"]
crime_df.to_csv(SUM_DIR / "crime_counts.csv", index=False)

# Static PNG (top 20)
plt.figure(figsize=(12, 7))
crime_counts.head(20).sort_values().plot(kind="barh", color="#667eea")
plt.title("Top 20 Crime Types", fontsize=14, fontweight="bold")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "top20_crime_types.png", dpi=150)
plt.close()

# Interactive Plotly
fig = px.bar(crime_df.head(25), x="Count", y="Primary Type", orientation="h",
             title="Crime Type Distribution (Top 25)", color="Count",
             color_continuous_scale="Viridis")
fig.update_layout(height=700, yaxis={"categoryorder": "total ascending"})
fig.write_html(FIG_DIR / "crime_type_distribution.html")
print("  ✅ Crime type charts saved")

# ── 2. Geographic Folium Heatmap ─────────────────────────
print("🗺️ Generating Folium geographic heatmap...")
latlon = df.dropna(subset=["Latitude", "Longitude"])
sample_latlon = latlon.sample(min(60_000, len(latlon)), random_state=42)
heat_data = list(zip(sample_latlon["Latitude"], sample_latlon["Longitude"]))

m = folium.Map(location=[41.8781, -87.6298], zoom_start=11,
               tiles="CartoDB dark_matter")
HeatMap(heat_data, radius=8, blur=12, min_opacity=0.4).add_to(m)
m.save(str(FIG_DIR / "crime_heatmap.html"))

# Static scatter
plt.figure(figsize=(9, 9))
plt.scatter(sample_latlon["Longitude"], sample_latlon["Latitude"],
            s=0.5, alpha=0.3, c="#667eea")
plt.title("Crime Locations (sample 60k)", fontsize=13, fontweight="bold")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(FIG_DIR / "geo_scatter.png", dpi=150)
plt.close()
print("  ✅ Geographic heatmap saved")

# ── 3. Temporal Trends ────────────────────────────────────
print("⏰ Temporal trend analysis...")

# Hourly distribution
hourly = df["Hour"].value_counts().sort_index()
hourly.to_csv(SUM_DIR / "hourly_counts.csv")
plt.figure(figsize=(12, 5))
plt.plot(hourly.index, hourly.values, marker="o", linewidth=2, color="#667eea")
plt.fill_between(hourly.index, hourly.values, alpha=0.15, color="#667eea")
plt.title("Crimes by Hour of Day", fontsize=13, fontweight="bold")
plt.xlabel("Hour (0-23)"); plt.ylabel("Count")
plt.xticks(range(0, 24)); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIG_DIR / "crimes_by_hour.png", dpi=150)
plt.close()

# Weekday × Hour heatmap matrix
if "Weekday" in df.columns and "Hour" in df.columns:
    hour_week = df.groupby(["Weekday", "Hour"]).size().unstack(fill_value=0)
    hour_week.to_csv(SUM_DIR / "hour_week_counts.csv")
    plt.figure(figsize=(14, 5))
    plt.imshow(hour_week.values, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="Crime Count")
    plt.title("Crime Intensity: Weekday × Hour", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day"); plt.ylabel("Weekday (0=Mon)")
    plt.xticks(range(0, 24), range(0, 24))
    plt.yticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "weekday_hour_heatmap.png", dpi=150)
    plt.close()

# Monthly trend
monthly = df.groupby(["Year", "Month"]).size().rename("count").reset_index()
monthly.to_csv(SUM_DIR / "monthly_trend.csv", index=False)
fig_m = px.line(monthly, x="Month", y="count", color="Year",
                title="Monthly Crime Counts by Year", markers=True)
fig_m.write_html(FIG_DIR / "monthly_trend_by_year.html")

# Seasonal
season_counts = df["Season"].value_counts() if "Season" in df.columns else pd.Series()
if not season_counts.empty:
    season_counts.to_csv(SUM_DIR / "season_counts.csv")
    plt.figure(figsize=(7, 5))
    colors = ["#667eea","#764ba2","#f093fb","#4facfe"]
    season_counts.plot(kind="bar", color=colors[:len(season_counts)])
    plt.title("Crimes by Season", fontsize=13, fontweight="bold")
    plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(FIG_DIR / "crimes_by_season.png", dpi=150)
    plt.close()
print("  ✅ Temporal charts saved")

# ── 4. Arrest Rate Analysis ───────────────────────────────
print("🚔 Arrest rate analysis...")
id_col = "ID" if "ID" in df.columns else "Date"
arrest_dom = df.groupby("Primary Type").agg(
    total=(id_col, "count"),
    arrests=("Arrest", "sum"),
    domestic=("Domestic", "sum")
).reset_index()
arrest_dom["arrest_rate"]   = arrest_dom["arrests"]  / arrest_dom["total"]
arrest_dom["domestic_rate"] = arrest_dom["domestic"] / arrest_dom["total"]
arrest_dom = arrest_dom.sort_values("total", ascending=False)
arrest_dom.to_csv(SUM_DIR / "arrest_domestic_by_type.csv", index=False)

plt.figure(figsize=(12, 7))
top15 = arrest_dom.head(15).set_index("Primary Type")
top15["arrest_rate"].sort_values().plot(kind="barh", color="#f093fb")
plt.title("Arrest Rate by Crime Type (Top 15)", fontsize=13, fontweight="bold")
plt.xlabel("Arrest Rate"); plt.tight_layout()
plt.savefig(FIG_DIR / "arrest_rate_by_type.png", dpi=150)
plt.close()
print("  ✅ Arrest rate chart saved")

# ── 5. Summary Statistics ─────────────────────────────────
summary_stats = df.describe(include="all").transpose()
summary_stats.to_csv(SUM_DIR / "general_summary_stats.csv")

if "Community Area" in df.columns:
    ca_top = df["Community Area"].value_counts().rename_axis("Community Area").reset_index(name="counts")
    ca_top.to_csv(SUM_DIR / "top_community_areas.csv", index=False)

# Small sample for UI drilldown
df.sample(2000, random_state=42).to_csv(SUM_DIR / "sample_for_ui.csv", index=False)

print(f"\n✅ EDA complete! Figures → {FIG_DIR}\n               Summaries → {SUM_DIR}")
