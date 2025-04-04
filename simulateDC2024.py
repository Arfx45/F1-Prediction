import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def prepare_historical_data():
    # Load historical data
    results_df = pd.read_csv('data/results.csv')
    races_df = pd.read_csv('data/races.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    circuits_df = pd.read_csv('data/circuits.csv')
    driver_elo_df = pd.read_csv('data/driver_elo.csv')

    # Filter for recent years (e.g., 2018-2023) for more relevant data
    races_df = races_df[races_df['year'].between(2018, 2023)]
    results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')
    
    # Create features for training
    merged_data = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
    merged_data['Driver Name'] = merged_data['forename'] + ' ' + merged_data['surname']
    
    return merged_data, races_df, circuits_df, driver_elo_df

def create_features(row, driver_elo_df, circuits_df):
    # Get driver Elo rating
    driver_name = row['Driver Name']
    driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver_name]['Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    # Get circuit features
    circuit_alt = circuits_df[circuits_df['circuitId'] == row['circuitId']]['alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0
    
    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': row['grid'] if row['grid'] != '\\N' else -1,
        'circuit_altitude': altitude,
        'qualifying_time': row['milliseconds'] if 'milliseconds' in row and row['milliseconds'] != '\\N' else 0
    })

def predict_2024_season():
    # Prepare data
    merged_data, races_df, circuits_df, driver_elo_df = prepare_historical_data()
    
    # Create features and target
    X = merged_data.apply(lambda x: create_features(x, driver_elo_df, circuits_df), axis=1)
    y = (merged_data['position'].astype(str).str.replace('\\N', '-1').astype(float) == 1).astype(int)
    
    # Train model on all historical data
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Load 2024 calendar
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    
    # Current F1 drivers for 2024
    current_drivers = [
        'Max Verstappen', 'Sergio Perez',
        'Lewis Hamilton', 'George Russell',
        'Charles Leclerc', 'Carlos Sainz',
        'Lando Norris', 'Oscar Piastri',
        'Fernando Alonso', 'Lance Stroll',
        'Pierre Gasly', 'Esteban Ocon',
        'Alexander Albon', 'Logan Sargeant',
        'Valtteri Bottas', 'Zhou Guanyu',
        'Daniel Ricciardo', 'Yuki Tsunoda',
        'Kevin Magnussen', 'Nico Hulkenberg'
    ]
    
    season_predictions = []
    
    # Predict each race
    for _, race in races_2024.iterrows():
        # Create race data
        race_data = []
        grid_positions = np.random.permutation(20) + 1  # Simulated qualifying
        
        for driver, grid in zip(current_drivers, grid_positions):
            driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver]['Elo Rating']
            elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
            
            race_data.append({
                'driver_elo': elo_rating,
                'grid_position': grid,
                'circuit_altitude': circuits_df[circuits_df['circuitId'] == race['circuitId']]['alt'].values[0],
                'qualifying_time': 0
            })
        
        # Make predictions
        race_df = pd.DataFrame(race_data)
        win_probabilities = model.predict_proba(race_df)[:, 1]
        
        # Store results
        results = pd.DataFrame({
            'Driver': current_drivers,
            'Win Probability': win_probabilities
        }).sort_values('Win Probability', ascending=False)
        
        season_predictions.append({
            'raceId': race['raceId'],
            'raceName': race['name'],
            'date': race['date'],
            'predicted_winner': results.iloc[0]['Driver'],
            'win_probability': results.iloc[0]['Win Probability'],
            'top_3': results.head(3)['Driver'].tolist()
        })
    
    # Save predictions
    predictions_df = pd.DataFrame(season_predictions)
    predictions_df.to_csv('data/2024/predicted_results2024DT.csv', index=False)
    
    return predictions_df

if __name__ == "__main__":
    predictions = predict_2024_season()
    
    # Display predictions
    print("\n2024 Season Predictions")
    print("=====================")
    
    # Calculate championship points
    points_system = {1: 25, 2: 18, 3: 15}
    championship_points = {}
    
    for _, race in predictions.iterrows():
        print(f"\n{race['raceName']} ({race['date']}):")
        print(f"Predicted Winner: {race['predicted_winner']} (P={race['win_probability']:.2f})")
        print(f"Predicted Podium: {', '.join(race['top_3'])}")
        
        # Update championship points
        for position, driver in enumerate(race['top_3'], 1):
            if position in points_system:
                championship_points[driver] = championship_points.get(driver, 0) + points_system[position]
    
    # Display predicted championship standings
    print("\nPredicted Championship Standings:")
    print("================================")
    standings = pd.DataFrame.from_dict(championship_points, orient='index', 
                                     columns=['Points']).sort_values('Points', ascending=False)
    print(standings)