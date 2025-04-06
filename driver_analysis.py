import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')

def analyze_driver_performance():
    print("Analyzing Driver Performance Patterns...")
    
    # Create output directory
    if not os.path.exists('analysis'):
        os.makedirs('analysis')
    
    # Merge driver names
    driver_results = results_df.merge(
        drivers_df[['driverId', 'forename', 'surname']], 
        on='driverId'
    )
    driver_results['Driver'] = driver_results['forename'] + ' ' + driver_results['surname']
    
    # Convert position to numeric
    driver_results['position'] = pd.to_numeric(driver_results['position'].replace('\\N', np.nan))
    
    # Calculate performance metrics
    performance = {
        'avg_position': driver_results.groupby('Driver')['position'].mean(),
        'wins': driver_results[driver_results['position'] == 1].groupby('Driver').size(),
        'podiums': driver_results[driver_results['position'] <= 3].groupby('Driver').size()
    }
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Top 10 drivers by average position
    plt.subplot(2, 1, 1)
    performance['avg_position'].sort_values().head(10).plot(kind='bar')
    plt.title('Top 10 Drivers by Average Position')
    plt.xticks(rotation=45)
    
    # Top 10 drivers by wins
    plt.subplot(2, 1, 2)
    performance['wins'].sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('Top 10 Drivers by Wins')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis/driver_performance.png')
    print("Driver analysis saved to analysis/driver_performance.png")

if __name__ == "__main__":
    analyze_driver_performance()