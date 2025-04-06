import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_race_results(results_df, races_df, drivers_df):
    # Create analysis directory if it doesn't exist
    if not os.path.exists('analysis'):
        os.makedirs('analysis')

    # Filter out 2024 data and keep only 2018-2023
    races_df = races_df[(races_df['year'] >= 2018) & (races_df['year'] < 2024)]
    
    # Merge datasets with filtered races
    merged_df = results_df.merge(races_df[['raceId', 'year', 'name']], on='raceId')
    merged_df = merged_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
    merged_df['Driver Name'] = merged_df['forename'] + ' ' + merged_df['surname']

    # Set the style for all plots
    sns.set_style("whitegrid")
    
    # 1. Podium Finishes Distribution
    plt.figure(figsize=(12, 6))
    podium_counts = merged_df[merged_df['position'].astype(str).isin(['1', '2', '3'])]['Driver Name'].value_counts()
    sns.barplot(x=podium_counts.head(10).index, y=podium_counts.head(10).values)
    plt.title('Top 10 Drivers by Podium Finishes (2018-2023)')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Podiums')
    plt.tight_layout()
    plt.savefig('analysis/podium_distribution.png')
    plt.close()

    # 2. Win Distribution
    plt.figure(figsize=(15, 8))
    wins = merged_df[merged_df['position'] == '1']['Driver Name'].value_counts()
    ax = sns.barplot(x=wins.index, y=wins.values)
    plt.title('Race Wins Distribution by Driver (2018-2023)', pad=20)
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(wins.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Driver Name', labelpad=15)
    plt.ylabel('Number of Wins', labelpad=15)
    plt.tight_layout()
    plt.savefig('analysis/wins_distribution.png')
    plt.close()

    # 3. Position Distribution Heatmap
    plt.figure(figsize=(12, 8))
    position_matrix = pd.crosstab(merged_df['Driver Name'], merged_df['position'])
    sns.heatmap(position_matrix.head(10), cmap='YlOrRd', annot=True, fmt='d')
    plt.title('Position Distribution Heatmap (Top 10 Drivers)')
    plt.tight_layout()
    plt.savefig('analysis/position_heatmap.png')
    plt.close()

    # 4. Year-wise Performance
    plt.figure(figsize=(12, 6))
    yearly_wins = pd.crosstab(merged_df['year'], merged_df['Driver Name'])[wins.head().index]
    sns.lineplot(data=yearly_wins)
    plt.title('Driver Performance Over Years (Top Winners)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('analysis/yearly_performance.png')
    plt.close()

    # Return summary statistics
    summary_stats = {
        'total_races': len(races_df),
        'unique_winners': len(wins),
        'most_wins': wins.head(1).to_dict(),
        'podium_appearances': podium_counts.head(5).to_dict()
    }
    
    return summary_stats

if __name__ == "__main__":
    # Load the data
    results_df = pd.read_csv('data/results.csv')
    races_df = pd.read_csv('data/races.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    
    # Run the analysis
    stats = analyze_race_results(results_df, races_df, drivers_df)
    
    # Print summary statistics
    print("\nRace Statistics Summary (2018-2023)")
    print("===================================")
    print(f"Total Races: {stats['total_races']}")
    print(f"Unique Winners: {stats['unique_winners']}")
    print("\nMost Wins:")
    for driver, wins in stats['most_wins'].items():
        print(f"{driver}: {wins} wins")
    print("\nTop 5 Podium Appearances:")
    for driver, podiums in stats['podium_appearances'].items():
        print(f"{driver}: {podiums} podiums")