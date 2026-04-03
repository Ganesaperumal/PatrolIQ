import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path: str, output_dir: str):
    print("Loading data for EDA...")
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Crime Type Distribution
    print("Generating crime type distribution...")
    plt.figure(figsize=(12, 8))
    top_crimes = df['Primary Type'].value_counts().head(20)
    sns.barplot(y=top_crimes.index, x=top_crimes.values, palette="viridis")
    plt.title('Top 20 Crime Types')
    plt.xlabel('Count')
    plt.ylabel('Crime Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crime_type_distribution.png'))
    plt.close()

    # 2. Hourly Trends
    print("Generating hourly trends...")
    plt.figure(figsize=(10, 6))
    hourly_counts = df.groupby('Hour').size()
    sns.lineplot(x=hourly_counts.index, y=hourly_counts.values, marker="o", color='b')
    plt.title('Crime Volume by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Number of Crimes')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_trends.png'))
    plt.close()
    
    # 3. Monthly Trends
    print("Generating monthly trends...")
    plt.figure(figsize=(10, 6))
    monthly_counts = df.groupby('Month').size()
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette="magma")
    plt.title('Crime Volume by Month')
    plt.xlabel('Month (1-12)')
    plt.ylabel('Number of Crimes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_trends.png'))
    plt.close()

    # 4. Arrest Rates
    print("Generating arrest rates distribution...")
    plt.figure(figsize=(6, 6))
    arrest_counts = df['Arrest'].value_counts()
    plt.pie(arrest_counts.values, labels=arrest_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title('Arrest Status Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'arrest_rates.png'))
    plt.close()

    print(f"EDA successfully completed. Visualizations saved into {output_dir}")

if __name__ == "__main__":
    input_file = "../data/processed/clean_crimes.csv"
    output_dir = "../data/figures"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_dir)
    
    run_eda(input_path, output_path)
