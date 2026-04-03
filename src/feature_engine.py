import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path: str, output_path: str):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")

    # 1. Coordinate Binning (Geographic Features)
    # Rounding latitude and longitude to 3 decimal places to create ~100m bins
    print("Creating coordinate bins...")
    df['Lat_Bin'] = df['Latitude'].round(3)
    df['Lon_Bin'] = df['Longitude'].round(3)

    # 2. Crime Severity Scores
    print("Developing crime severity scores...")
    high_severity = ['HOMICIDE', 'CRIM SEXUAL ASSAULT', 'ROBBERY', 'BATTERY', 'ASSAULT', 'KIDNAPPING', 'ARSON']
    medium_severity = ['BURGLARY', 'MOTOR VEHICLE THEFT', 'THEFT', 'WEAPONS VIOLATION', 'INTIMIDATION', 'CRIMINAL DAMAGE']
    
    def get_severity(crime_type):
        if crime_type in high_severity:
            return 3
        elif crime_type in medium_severity:
            return 2
        else:
            return 1
            
    df['Crime_Severity_Score'] = df['Primary Type'].apply(get_severity)

    # 3. Categorical Encoding
    # Since there are many categories, we might use Frequency Encoding so we don't blow up the dimensionality
    print("Applying categorical encoding...")
    type_freq = df['Primary Type'].value_counts(normalize=True)
    df['Primary_Type_Freq_Enc'] = df['Primary Type'].map(type_freq)
    
    loc_freq = df['Location Description'].value_counts(normalize=True)
    df['Location_Desc_Freq_Enc'] = df['Location Description'].map(loc_freq)

    # Label encode Arrest and Domestic boolean flags
    df['Arrest_Encoded'] = df['Arrest'].astype(int)
    df['Domestic_Encoded'] = df['Domestic'].astype(int)

    # Temporally encode cyclical features using Sine/Cosine mapping for Hour and Month
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24.0)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12.0)

    # 4. Normalize Geographic Coordinates
    print("Normalizing geographic coordinates...")
    scaler = StandardScaler()
    df[['Latitude_Scaled', 'Longitude_Scaled']] = scaler.fit_transform(df[['Latitude', 'Longitude']])

    print(f"Final shape before saving: {df.shape}")
    
    # Save model-ready data
    print(f"Saving model-ready data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Feature engineering successfully completed!")

if __name__ == "__main__":
    input_file = "../data/processed/clean_crimes.csv"
    output_file = "../data/processed/model_ready_crimes.csv"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_file)
    
    engineer_features(input_path, output_path)
