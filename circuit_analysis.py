import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
results_df = pd.read_csv('data/results.csv')
circuits_df = pd.read_csv('data/circuits.csv')
races_df = pd.read_csv('data/races.csv')

def analyze_circuits():
    print("Analyzing Circuit Characteristics...")
    
    # Create output directory
    if not os.path.exists('analysis'):
        os.makedirs('analysis')
    
    # Merge race results with circuit data
    circuit_results = results_df.merge(races_df[['raceId', 'circuitId']], on='raceId')
    circuit_results = circuit_results.merge(circuits_df, on='circuitId')
    
    # Calculate circuit statistics
    circuit_stats = {
        'wins_distribution': circuit_results[circuit_results['position'] == '1'].groupby('name').size(),
        'avg_finishers': circuit_results.groupby('name').size() / races_df.groupby('circuitId').size(),
        'altitude_effect': pd.DataFrame({
            'altitude': circuits_df['alt'],
            'wins': circuit_results[circuit_results['position'] == '1'].groupby('circuitId').size()
        })
    }
    
    # Create visualizations
    plt.figure(figsize=(15, 15))
    
    # Circuit win distribution
    plt.subplot(3, 1, 1)
    circuit_stats['wins_distribution'].sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('Top 10 Circuits by Number of Different Winners')
    plt.xticks(rotation=45)
    
    # Altitude vs Performance
    plt.subplot(3, 1, 2)
    sns.scatterplot(data=circuit_stats['altitude_effect'], x='altitude', y='wins')
    plt.title('Circuit Altitude vs Number of Wins')
    
    # Average finishers per circuit
    plt.subplot(3, 1, 3)
    circuit_stats['avg_finishers'].sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('Top 10 Circuits by Average Number of Finishers')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis/circuit_characteristics.png')
    print("Circuit analysis saved to analysis/circuit_characteristics.png")

if __name__ == "__main__":
    analyze_circuits()