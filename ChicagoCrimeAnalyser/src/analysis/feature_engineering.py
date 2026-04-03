"""
PatrolIQ - Advanced Feature Engineering
Best-of-both: Cyclic sin/cos + crime severity + frequency encoding + label encoding + scaler save
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_PATH  = BASE_DIR / "data" / "processed" / "sample_500000_rows.csv"
OUT_PATH   = BASE_DIR / "data" / "processed" / "model_ready_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("📂 Loading cleaned data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
print(f"✅ Loaded: {df.shape}")

# 1. Cyclic Temporal Encoding (mathematically preserves circular nature)
print("🔄 Cyclic temporal encoding...")
df["Hour_Sin"]  = np.sin(2 * np.pi * df["Hour"] / 24.0)
df["Hour_Cos"]  = np.cos(2 * np.pi * df["Hour"] / 24.0)
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
df["Day_Sin"]   = np.sin(2 * np.pi * df["Weekday"] / 7.0)
df["Day_Cos"]   = np.cos(2 * np.pi * df["Weekday"] / 7.0)

# 2. Crime Severity Score
print("⚠️ Building crime severity scores...")
high_severity   = {"HOMICIDE","CRIM SEXUAL ASSAULT","CRIMINAL SEXUAL ASSAULT",
                   "ROBBERY","BATTERY","ASSAULT","KIDNAPPING","ARSON","HUMAN TRAFFICKING"}
medium_severity = {"BURGLARY","MOTOR VEHICLE THEFT","THEFT","WEAPONS VIOLATION",
                   "INTIMIDATION","CRIMINAL DAMAGE","STALKING","SEX OFFENSE"}

def get_severity(ctype):
    if ctype in high_severity:   return 3
    elif ctype in medium_severity: return 2
    return 1

df["Crime_Severity_Score"] = df["Primary Type"].apply(get_severity)

# 3. Frequency Encoding for high-cardinality categoricals
print("📊 Frequency encoding...")
for col, new_col in [("Primary Type",         "Primary_Type_Freq_Enc"),
                     ("Location Description", "Location_Desc_Freq_Enc")]:
    if col in df.columns:
        freq = df[col].value_counts(normalize=True)
        df[new_col] = df[col].map(freq)

# 4. Label Encoding for ML-usable versions of categoricals
print("🏷️ Label encoding...")
le_crime    = LabelEncoder()
le_location = LabelEncoder()
le_season   = LabelEncoder()
le_time     = LabelEncoder()

if "Primary Type"         in df.columns: df["Crime_Label"]    = le_crime.fit_transform(df["Primary Type"].astype(str))
if "Location Description" in df.columns: df["Location_Label"] = le_location.fit_transform(df["Location Description"].astype(str))
if "Season"               in df.columns: df["Season_Label"]   = le_season.fit_transform(df["Season"].astype(str))

# Time of day bucket
def time_of_day(h):
    if 5  <= h < 12: return "Morning"
    elif 12 <= h < 17: return "Afternoon"
    elif 17 <= h < 21: return "Evening"
    else:             return "Night"

df["Time_of_Day"] = df["Hour"].apply(time_of_day)
df["Time_Label"]  = le_time.fit_transform(df["Time_of_Day"])

# 5. Boolean encoding
df["Arrest_Enc"]   = df["Arrest"].astype(int)
df["Domestic_Enc"] = df["Domestic"].astype(int)

# 6. Coordinate binning (~100m grid cells)
df["Lat_Bin"] = df["Latitude"].round(3)
df["Lon_Bin"] = df["Longitude"].round(3)

# 7. Normalize geographic coordinates and save scaler
print("📐 Scaling lat/lon...")
scaler = StandardScaler()
df[["Latitude_Scaled", "Longitude_Scaled"]] = scaler.fit_transform(df[["Latitude","Longitude"]])
joblib.dump(scaler, MODELS_DIR / "latlon_scaler.pkl")

# Save label encoders
joblib.dump(le_crime,    MODELS_DIR / "label_crime.pkl")
joblib.dump(le_location, MODELS_DIR / "label_location.pkl")
joblib.dump(le_season,   MODELS_DIR / "label_season.pkl")
joblib.dump(le_time,     MODELS_DIR / "label_time.pkl")

print(f"✅ Final shape: {df.shape}")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"✅ Model-ready data saved to: {OUT_PATH}")
