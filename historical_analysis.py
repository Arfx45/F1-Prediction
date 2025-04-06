import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')

def analyze_historical_trends():
    print("Analyzing Historical Trends...")
    
    # Create output directory
    if not os.path.exists('analysis'):
        os.makedirs('analysis')
    
    # Merge race data with results
    historical_data = results_df.merge(races_df[['raceId', 'year']], on='raceId')
    
    # Calculate yearly statistics
    yearly_stats = {
        'winners': historical_data[historical_data['position'] == '1'].groupby('year').size(),
        'avg_finishers': historical_data.groupby('year').size() / races_df.groupby('year').size(),
        'competitive_index': historical_data.groupby('year')['position'].nunique()
    }
    
    # Create visualizations
    plt.figure(figsize=(15, 15))
    
    # Number of different winners per year
    plt.subplot(3, 1, 1)
    yearly_stats['winners'].plot(kind='line', marker='o')
    plt.title('Number of Different Winners per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Winners')
    
    # Average finishers per race by year
    plt.subplot(3, 1, 2)
    yearly_stats['avg_finishers'].plot(kind='line', marker='o')
    plt.title('Average Number of Finishers per Race by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Finishers')
    
    # Competitiveness index (number of different positions achieved)
    plt.subplot(3, 1, 3)
    yearly_stats['competitive_index'].plot(kind='line', marker='o')
    plt.title('Competition Index by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Different Positions')
    
    plt.tight_layout()
    plt.savefig('analysis/historical_trends.png')
    print("Historical analysis saved to analysis/historical_trends.png")

if __name__ == "__main__":
    analyze_historical_trends()