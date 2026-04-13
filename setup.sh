#!/bin/bash
# PatrolIQ — Initial Setup Script
# Run this once after cloning the repo on Streamlit Cloud or locally

echo "🚔 PatrolIQ Setup Starting..."

# Create required directories
mkdir -p data/uncleaned data/cleaned

# Check if raw data exists
if [ ! -f "data/uncleaned/crimes_data.csv" ]; then
    echo "⚠️  Raw data not found at data/uncleaned/crimes_data.csv"
    echo "    Please upload your crimes_data.csv to data/uncleaned/"
    echo "    Download from: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2"
    exit 1
fi

# Run preprocessing if cleaned data doesn't exist
if [ ! -f "data/cleaned/cleaned_crimes.csv" ]; then
    echo "📊 Running preprocessing..."
    python3 scripts/preprocess.py
fi

# Run training if ML artifacts don't exist
if [ ! -f "data/cleaned/tsne_result.csv" ]; then
    echo "🤖 Running ML training pipeline (this takes 5-10 minutes)..."
    python3 scripts/train.py
fi

echo "✅ Setup complete! Run: streamlit run app.py"
