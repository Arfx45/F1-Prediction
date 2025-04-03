import pandas as pd
import numpy as np
from collections import defaultdict

# Load the data
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')

# Filter out 2024 races
races_df = races_df[races_df['year'] != 2024]

# Merge results with filtered race information
merged_df = pd.merge(results_df, races_df[['raceId', 'year']], on='raceId')

# Sort by raceId to ensure chronological order
merged_df = merged_df.sort_values('raceId')

# Elo system initialization
K = 32  # K-factor
initial_rating = 1500
driver_ratings = defaultdict(lambda: initial_rating)
peak_elo = defaultdict(lambda: initial_rating)  # New dictionary to track peak Elo
last_race_year = {}  # Track the last year each driver participated

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(winner_rating, loser_rating):
    expected_winner = calculate_expected_score(winner_rating, loser_rating)
    new_winner_rating = winner_rating + K * (1 - expected_winner)
    new_loser_rating = loser_rating + K * (0 - (1 - expected_winner))
    return new_winner_rating, new_loser_rating

# Process races and update Elo ratings
for race_id, race_group in merged_df.groupby('raceId'):
    race_year = race_group['year'].iloc[0]  # Get the year of the race

    for _, driver1 in race_group.iterrows():
        if pd.isna(driver1['position']) or driver1['position'] == '\\N':
            continue
        
        position1 = int(driver1['position'])
        driver_id1 = driver1['driverId']
        last_race_year[driver_id1] = race_year  # Update last active year
        
        for _, driver2 in race_group.iterrows():
            if pd.isna(driver2['position']) or driver2['position'] == '\\N' or driver1['driverId'] == driver2['driverId']:
                continue
            
            position2 = int(driver2['position'])
            driver_id2 = driver2['driverId']
            last_race_year[driver_id2] = race_year  # Update last active year
            
            if position1 < position2:
                driver_ratings[driver_id1], driver_ratings[driver_id2] = update_elo(
                    driver_ratings[driver_id1], driver_ratings[driver_id2]
                )

                # Update peak Elo if the new Elo is the highest recorded
                peak_elo[driver_id1] = max(peak_elo[driver_id1], driver_ratings[driver_id1])
                peak_elo[driver_id2] = max(peak_elo[driver_id2], driver_ratings[driver_id2])

# Apply Elo decay for inactive drivers
current_year = 2024
decay_rate = 15  # Elo points lost per inactive year

for driver_id in driver_ratings.keys():
    if driver_id in last_race_year:
        years_inactive = max(0, current_year - last_race_year[driver_id])
        driver_ratings[driver_id] -= years_inactive * decay_rate
        driver_ratings[driver_id] = max(1400, driver_ratings[driver_id])  # Prevent dropping too low

# Convert final ratings to a DataFrame
final_ratings = pd.DataFrame.from_dict(driver_ratings, orient='index', columns=['Elo Rating'])
final_ratings.index.name = 'driverId'
final_ratings.reset_index(inplace=True)

# Merge peak Elo values
final_ratings['Peak Elo'] = final_ratings['driverId'].map(peak_elo)

# Merge with driver names
final_ratings = final_ratings.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
final_ratings['Driver Name'] = final_ratings['forename'] + ' ' + final_ratings['surname']

# Organize columns and sort by current Elo
final_ratings = final_ratings[['Driver Name', 'Elo Rating', 'Peak Elo']].sort_values('Elo Rating', ascending=False)

print("Top 10 Drivers by Elo Rating:")
print(final_ratings.head(10))

# Export to CSV
final_ratings.to_csv('data/driver_elo.csv', index=False)
print("Driver Elo ratings have been saved to 'data/driver_elo.csv'")
