"""
PatrolIQ — Data Preprocessing Pipeline
Cleans the raw Chicago crime dataset and engineers features for ML.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

RAW_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "uncleaned", "crimes_data.csv")
CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
os.makedirs(CLEAN_DIR, exist_ok=True)

# ── Severity map for 33 crime types ──────────────────────────────────────────
SEVERITY_MAP = {
    "HOMICIDE": 10, "KIDNAPPING": 9, "HUMAN TRAFFICKING": 9,
    "CRIMINAL SEXUAL ASSAULT": 8, "SEX OFFENSE": 7, "ROBBERY": 7,
    "ARSON": 7, "ASSAULT": 6, "BATTERY": 6, "STALKING": 6,
    "INTIMIDATION": 5, "BURGLARY": 5, "MOTOR VEHICLE THEFT": 5,
    "WEAPONS VIOLATION": 5, "OFFENSE INVOLVING CHILDREN": 5,
    "NARCOTICS": 4, "OTHER NARCOTIC VIOLATION": 4, "PROSTITUTION": 4,
    "CRIMINAL DAMAGE": 3, "CRIMINAL TRESPASS": 3, "DECEPTIVE PRACTICE": 3,
    "THEFT": 3, "FRAUD": 3, "FORGERY & COUNTERFEITING": 3,
    "INTERFERENCE WITH PUBLIC OFFICER": 2, "LIQUOR LAW VIOLATION": 2,
    "GAMBLING": 2, "PUBLIC PEACE VIOLATION": 2, "OBSCENITY": 2,
    "CRIM SEXUAL ASSAULT": 8, "NON-CRIMINAL": 1, "NON - CRIMINAL": 1,
    "OTHER OFFENSE": 2, "CONCEALED CARRY LICENSE VIOLATION": 2,
}

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring",  4: "Spring", 5: "Spring",
              6: "Summer",  7: "Summer", 8: "Summer",
              9: "Fall",   10: "Fall",  11: "Fall"}

DAYNUM_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
              "Friday": 4, "Saturday": 5, "Sunday": 6}

def load_raw(path: str, nrows: int = None) -> pd.DataFrame:
    print(f"[preprocess] Loading raw CSV …")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    print(f"[preprocess] Raw shape: {df.shape}")
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing critical columns
    critical_cols = ["Latitude", "Longitude", "Primary Type", "Date"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"[preprocess] Dropped {before - len(df)} rows with missing geo/date/type  → {len(df)}")

    # Remove obvious geo outliers (Chicago bounding box)
    df = df[(df["Latitude"].between(41.5, 42.1)) & (df["Longitude"].between(-87.95, -87.4))]
    print(f"[preprocess] After geo filter: {len(df)}")

    # Parse datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date"])

    # ── Feature Engineering ────────────────────────────────────────────────
    df["Hour"]        = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Day_Num"]     = df["Day_of_Week"].map(DAYNUM_MAP).fillna(0).astype(int)
    df["Month"]       = df["Date"].dt.month
    df["Year"]        = df["Date"].dt.year
    df["Season"]      = df["Month"].map(SEASON_MAP)
    df["Is_Weekend"]  = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)
    df["Is_Night"]    = ((df["Hour"] >= 20) | (df["Hour"] < 6)).astype(int)
    df["Crime_Severity_Score"] = df["Primary Type"].map(SEVERITY_MAP).fillna(2).astype(int)

    # Encode Arrest / Domestic as int
    for col in ["Arrest", "Domestic"]:
        if df[col].dtype == object:
            df[col] = df[col].str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
        else:
            df[col] = df[col].astype(int)

    # District / Beat as int
    for col in ["District", "Beat", "Ward", "Community Area"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    print(f"[preprocess] Feature engineering done. Final shape: {df.shape}")
    return df

def save_metadata(df: pd.DataFrame):
    meta = {
        "total_records": int(len(df)),
        "crime_types":   sorted(df["Primary Type"].unique().tolist()),
        "years":         sorted(df["Year"].unique().tolist()),
        "districts":     sorted(df["District"].unique().tolist()),
        "arrest_rate":   float(round(df["Arrest"].mean() * 100, 2)),
        "domestic_rate": float(round(df["Domestic"].mean() * 100, 2)),
        "columns":       df.columns.tolist(),
    }
    with open(os.path.join(CLEAN_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[preprocess] metadata.json saved with {len(meta['crime_types'])} crime types.")

def main():
    df = load_raw(RAW_PATH)
    df = clean(df)
    save_metadata(df)

    out_path = os.path.join(CLEAN_DIR, "cleaned_crimes.csv")
    df.to_csv(out_path, index=False)
    print(f"[preprocess] ✅ Saved → {out_path}")
    print(f"[preprocess] Shape: {df.shape}, Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()
