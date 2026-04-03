"""
PatrolIQ - Data Validation & Quality Report
Produces both a text report and JSON summary.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from io import StringIO

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "sample_500000_rows.csv"
REPORT_DIR = BASE_DIR / "reports" / "summaries"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def validate_data(data_path: Path) -> dict:
    buffer = StringIO()
    def log(msg):
        print(msg); buffer.write(msg + "\n")

    log("🔍 Starting Data Quality Validation")
    log("-" * 60)
    summary = {}

    try:
        df = pd.read_csv(data_path, low_memory=False)
        log(f"✅ Loaded: {len(df):,} records, {len(df.columns)} columns")
        summary["records"] = len(df)
        summary["columns"] = len(df.columns)
    except Exception as e:
        log(f"❌ Error loading data: {e}"); return {}

    # Missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    summary["missing_values"] = missing_cols.to_dict()
    if not missing_cols.empty:
        log(f"\n🧩 Missing values found:\n{missing_cols}")
    else:
        log("\n✅ No missing values detected")

    # Duplicates
    dup_count = df.duplicated().sum()
    log(f"\n📦 Duplicate rows: {dup_count}")
    summary["duplicates"] = int(dup_count)

    # Date validation
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = df[df["Date"] > pd.Timestamp.today()]
        log(f"\n⏰ Future dates found: {len(invalid_dates)}")
        log(f"📅 Date range: {df['Date'].min()} → {df['Date'].max()}")
        summary["date_range"] = [str(df["Date"].min()), str(df["Date"].max())]

    # Year range
    if "Year" in df.columns:
        log(f"📆 Year range: {int(df['Year'].min())} → {int(df['Year'].max())}")
        summary["year_range"] = [int(df["Year"].min()), int(df["Year"].max())]

    # Geographic validation
    if {"Latitude", "Longitude"}.issubset(df.columns):
        geo_outliers = df[(df["Latitude"] < 41.6) | (df["Latitude"] > 42.1) |
                         (df["Longitude"] < -87.95) | (df["Longitude"] > -87.5)]
        log(f"\n🗺️ Out-of-bound coordinates: {len(geo_outliers)}")
        summary["geo_outliers"] = int(len(geo_outliers))

    # Crime category validation
    known_crimes = [
        'THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT', 'BURGLARY',
        'MOTOR VEHICLE THEFT', 'ROBBERY', 'DECEPTIVE PRACTICE', 'CRIMINAL TRESPASS',
        'WEAPONS VIOLATION', 'PUBLIC PEACE VIOLATION', 'OFFENSE INVOLVING CHILDREN',
        'CRIM SEXUAL ASSAULT', 'SEX OFFENSE', 'GAMBLING', 'LIQUOR LAW VIOLATION',
        'ARSON', 'INTERFERENCE WITH PUBLIC OFFICER', 'HOMICIDE', 'KIDNAPPING',
        'INTIMIDATION', 'STALKING', 'OBSCENITY', 'OTHER OFFENSE', 'OTHER NARCOTIC VIOLATION',
        'NON-CRIMINAL', 'PROSTITUTION', 'HUMAN TRAFFICKING', 'PUBLIC INDECENCY',
        'CONCEALED CARRY LICENSE VIOLATION', 'CRIMINAL SEXUAL ASSAULT'
    ]
    if "Primary Type" in df.columns:
        invalid_types = df[~df["Primary Type"].isin(known_crimes)]
        log(f"🚫 Unlisted crime categories: {len(invalid_types)}")
        log(f"📊 Unique crime types found: {df['Primary Type'].nunique()}")
        summary["unique_crime_types"] = int(df["Primary Type"].nunique())
        summary["unlisted_crime_types"] = int(len(invalid_types))

    # Temporal coverage
    for col in ["Hour", "Month", "Day"]:
        if col in df.columns:
            summary[f"{col.lower()}_range"] = [int(df[col].min()), int(df[col].max())]

    # Arrest & case number integrity
    if {"Arrest", "Case Number"}.issubset(df.columns):
        invalid_arrests = df[(df["Arrest"] == True) & (df["Case Number"].isna())]
        log(f"\n⚠️ Arrests without Case Number: {len(invalid_arrests)}")
        summary["invalid_arrests"] = int(len(invalid_arrests))

    log("\n✅ Data quality validation completed!")
    log("-" * 60)

    # Save reports
    txt_path  = REPORT_DIR / "data_validation_report.txt"
    json_path = REPORT_DIR / "data_validation_summary.json"
    with open(txt_path, "w")  as f: f.write(buffer.getvalue())
    with open(json_path, "w") as f: json.dump(summary, f, indent=4)
    print(f"📝 Reports saved to: {REPORT_DIR}")
    return summary

if __name__ == "__main__":
    validate_data(DATA_PATH)
