import pandas as pd
import numpy as np
import os
import sys

def load_and_clean_data(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    print(f"Initial shape: {df.shape}")
    
    # Drop rows with missing crucial location data
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Fill remaining missing values for some categorical columns if any
    df['Location Description'].fillna('UNKNOWN', inplace=True)
    df['Ward'].fillna(-1, inplace=True)
    df['Community Area'].fillna(-1, inplace=True)
    
    print(f"Shape after dropping missing lat/lon: {df.shape}")
    
    # Extract temporal features
    print("Extracting temporal features...")
    df['Date_Parsed'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Drop rows where Date_Parsed couldn't be parsed
    df = df.dropna(subset=['Date_Parsed'])
    
    df['Hour'] = df['Date_Parsed'].dt.hour
    df['Day_of_Week'] = df['Date_Parsed'].dt.day_name()
    df['Month'] = df['Date_Parsed'].dt.month
    df['Year_Extracted'] = df['Date_Parsed'].dt.year
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday'])
    
    # Define seasons: Winter (12,1,2), Spring (3,4,5), Summer (6,7,8), Fall (9,10,11)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
            
    df['Season'] = df['Month'].apply(get_season)
    
    # Drop the temporary Date_Parsed and other unnecessary columns if we want, or keep them
    df.drop(columns=['Date_Parsed'], inplace=True)
    
    # Data Quality Check
    print("Running Data Quality Checks...")
    numeric_outliers = (df['Latitude'] < 41.6) | (df['Latitude'] > 42.1) | (df['Longitude'] < -88.0) | (df['Longitude'] > -87.5)
    if numeric_outliers.any():
        print(f"Found {numeric_outliers.sum()} anomalous coordinates. Filtering them out.")
        df = df[~numeric_outliers]
    
    print(f"Final shape before saving: {df.shape}")
    
    # Save the processed dataset
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    INPUT_FILE = "../data/unprocessed/Crimes_2001_to_Present_20260401.csv"
    OUTPUT_FILE = "../data/processed/clean_crimes.csv"
    
    # Calculate path relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    
    load_and_clean_data(input_path, output_path)
