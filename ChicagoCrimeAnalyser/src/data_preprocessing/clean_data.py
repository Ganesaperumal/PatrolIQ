"""
PatrolIQ - Data Cleaning & Stratified Sampling
Best-of-both: stratified time-group sampling + IQR clipping + full temporal features
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "unprocessed" / "Crimes_2001_to_Present_20260401.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "sample_500000_rows.csv"

print(f"📂 Loading dataset from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"✅ Raw shape: {df.shape}")

# 1. Drop fully null rows & duplicates
df = df.dropna(subset=["Latitude", "Longitude", "Date"]).drop_duplicates(keep="last")
print(f"✅ After dropping NaN lat/lon/date & duplicates: {df.shape}")

# 2. Parse dates and extract temporal features
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df[df["Date"].dt.year >= 2010]

df["Year"]    = df["Date"].dt.year
df["Month"]   = df["Date"].dt.month
df["Day"]     = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday   # 0=Mon
df["Hour"]    = df["Date"].dt.hour
df["Minute"]  = df["Date"].dt.minute
df["Is_Weekend"] = df["Weekday"].isin([5, 6]).astype(int)

season_map = {12:"Winter",1:"Winter",2:"Winter",
              3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",
              9:"Fall",10:"Fall",11:"Fall"}
df["Season"] = df["Month"].map(season_map)

# 3. Fill remaining categoricals
df["Location Description"] = df["Location Description"].fillna("UNKNOWN")
df["Ward"]           = df["Ward"].fillna(-1)
df["Community Area"] = df["Community Area"].fillna(-1)

# 4. IQR clipping on numeric columns (removes extreme outliers)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude ID-like / encoded columns from clipping
skip_clip = ["Year", "Month", "Day", "Weekday", "Hour", "Minute", "Is_Weekend",
             "Ward", "Community Area", "Beat", "District", "X Coordinate", "Y Coordinate"]
for col in num_cols:
    if col in skip_clip:
        continue
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# 5. Geographic bounding box filter for Chicago
df = df[(df["Latitude"] >= 41.6) & (df["Latitude"] <= 42.1) &
        (df["Longitude"] >= -87.95) & (df["Longitude"] <= -87.5)]
print(f"✅ After geo bounding box filter: {df.shape}")

# 6. Stratified sampling — ensures temporal diversity across the dataset
print("🔀 Performing stratified time-group sampling...")
time_groups = df.groupby(["Year","Month","Day","Hour"], group_keys=False)
core_sample = time_groups.apply(lambda x: x.sample(n=1, random_state=42))
print(f"  Core sample size: {len(core_sample)}")

remaining_needed = 500_000 - len(core_sample)
if remaining_needed > 0:
    remaining_df  = df.drop(core_sample.index)
    n_extra = min(remaining_needed, len(remaining_df))
    extra_sample  = remaining_df.sample(n=n_extra, random_state=42)
    final_sample  = pd.concat([core_sample, extra_sample]).reset_index(drop=True)
else:
    final_sample = core_sample.sample(500_000, random_state=42).reset_index(drop=True)

print(f"✅ Final sample size: {len(final_sample)}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final_sample.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved to: {OUTPUT_PATH}")
