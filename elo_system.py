import pandas as pd
import numpy as np
from collections import defaultdict

# Load the data
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')

# Filter out 2024 races
races_df = races_df[races_df['year'] != 2024]

# Merge results with filtered race information
merged_df = pd.merge(results_df, races_df[['raceId', 'year']], on='raceId')

# Sort by raceId to ensure chronological order
merged_df = merged_df.sort_values('raceId')

# Initialize Elo ratings
K = 32  # K-factor
initial_rating = 1500
driver_ratings = defaultdict(lambda: initial_rating)

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(winner_rating, loser_rating):
    expected_winner = calculate_expected_score(winner_rating, loser_rating)
    new_winner_rating = winner_rating + K * (1 - expected_winner)
    new_loser_rating = loser_rating + K * (0 - (1 - expected_winner))
    return new_winner_rating, new_loser_rating

# Process races and update Elo ratings
for race_id, race_group in merged_df.groupby('raceId'):
    for _, driver1 in race_group.iterrows():
        # Skip if position is NaN or '\\N'
        if pd.isna(driver1['position']) or driver1['position'] == '\\N':
            continue
        
        position1 = int(driver1['position'])
        driver_id1 = driver1['driverId']
        
        for _, driver2 in race_group.iterrows():
            # Skip if position is NaN or '\\N' or same driver
            if pd.isna(driver2['position']) or driver2['position'] == '\\N' or driver1['driverId'] == driver2['driverId']:
                continue
            
            position2 = int(driver2['position'])
            driver_id2 = driver2['driverId']
            
            if position1 < position2:
                driver_ratings[driver_id1], driver_ratings[driver_id2] = update_elo(
                    driver_ratings[driver_id1], driver_ratings[driver_id2]
                )

# Convert final ratings to a DataFrame for analysis
final_ratings = pd.DataFrame.from_dict(driver_ratings, orient='index', columns=['Elo Rating'])
final_ratings = final_ratings.sort_values('Elo Rating', ascending=False)

# Merge with driver names for better readability
drivers_df = pd.read_csv('data/drivers.csv')
final_ratings = final_ratings.merge(drivers_df[['driverId', 'forename', 'surname']], left_index=True, right_on='driverId')
final_ratings['Driver Name'] = final_ratings['forename'] + ' ' + final_ratings['surname']
final_ratings = final_ratings[['Driver Name', 'Elo Rating']].sort_values('Elo Rating', ascending=False)

print("Top 10 Drivers by Elo Rating:")
print(final_ratings.head(10))

# Export to CSV
final_ratings.to_csv('data/driver_elo.csv', index=False)
print("Driver Elo ratings have been saved to 'data/driver_elo.csv'")