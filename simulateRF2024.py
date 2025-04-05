import pandas as pd
import numpy as np
import joblib
import os

def load_rf_model():
    model_path = os.path.join('__pycache__', 'randforestF12024.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError("Trained model file not found. Please run randomforest.py first.")

def simulate_2024_season():
    # Load the trained model
    rf_model = load_rf_model()
    
    # Load 2024 data
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    drivers_2024 = pd.read_csv('data/2024/drivers2024.csv')
    
    # Current F1 drivers for 2024
    current_drivers = [
        'Max Verstappen', 'Sergio Perez',  # Red Bull
        'Lewis Hamilton', 'George Russell',  # Mercedes
        'Charles Leclerc', 'Carlos Sainz',  # Ferrari
        'Lando Norris', 'Oscar Piastri',  # McLaren
        'Fernando Alonso', 'Lance Stroll',  # Aston Martin
        'Pierre Gasly', 'Esteban Ocon',  # Alpine
        'Alexander Albon', 'Logan Sargeant',  # Williams
        'Valtteri Bottas', 'Zhou Guanyu',  # Sauber
        'Daniel Ricciardo', 'Yuki Tsunoda',  # RB
        'Kevin Magnussen', 'Nico Hulkenberg'  # Haas
    ]

    # Store season results
    season_results = []

    # Load required data for predictions
    drivers_df = pd.read_csv('data/drivers.csv')
    circuits_df = pd.read_csv('data/circuits.csv')
    driver_elo_df = pd.read_csv('data/driver_elo.csv')
    lap_times_df = pd.read_csv('data/lap_times.csv')

    # Create a driver lookup dictionary
    driver_lookup = {}
    for driver in current_drivers:
        driver_match = drivers_df[
            (drivers_df['forename'] + ' ' + drivers_df['surname']) == driver
        ]
        if not driver_match.empty:
            driver_lookup[driver] = driver_match['driverId'].iloc[0]
        else:
            print(f"Warning: No driverId found for {driver}")
            driver_lookup[driver] = -1

    # Simulate each race
    for _, race in races_2024.iterrows():
        # Simulate qualifying (random grid positions for simplicity)
        grid_positions = np.random.permutation(20) + 1
        
        # Prepare race data
        race_data = []
        for i, (driver, grid) in enumerate(zip(current_drivers, grid_positions)):
            # Get driver Elo rating
            driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver]['Elo Rating']
            elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
            
            # Get circuit altitude
            circuit_alt = circuits_df[circuits_df['circuitId'] == race['circuitId']]['alt'].values[0]
            
            # Get driver's best lap time using lookup dictionary
            driver_id = driver_lookup[driver]
            if driver_id != -1:
                best_lap = lap_times_df[
                    lap_times_df['driverId'] == driver_id
                ]['milliseconds'].min()
            else:
                best_lap = lap_times_df['milliseconds'].mean()
            
            race_data.append({
                'driver_elo': elo_rating,
                'grid_position': grid,
                'circuit_altitude': circuit_alt,
                'qualifying_time': best_lap if pd.notna(best_lap) else 0
            })
        
        # Convert to DataFrame and make predictions
        race_df = pd.DataFrame(race_data)
        win_probabilities = rf_model.predict_proba(race_df)[:, 1]
        
        # Create results DataFrame
        predictions = pd.DataFrame({
            'Driver': current_drivers,
            'Win Probability': win_probabilities
        }).sort_values('Win Probability', ascending=False)
        
        # Store race results
        race_result = {
            'raceId': race['raceId'],
            'raceName': race['name'],
            'date': race['date'],
            'winner': predictions.iloc[0]['Driver'],
            'winProbability': predictions.iloc[0]['Win Probability'],
            'topThree': predictions.head(3)['Driver'].tolist()
        }
        season_results.append(race_result)
    
    # Convert to DataFrame and save results
    season_df = pd.DataFrame(season_results)
    season_df.to_csv('data/2024/simulated_results2024RF.csv', index=False)
    
    # Print season summary
    print("\n2024 Season Simulation Summary")
    print("==============================")
    print(f"Total races simulated: {len(season_results)}")
    
    # Calculate championship points (simplified: 25 for win)
    championship_points = {}
    for result in season_results:
        winner = result['winner']
        championship_points[winner] = championship_points.get(winner, 0) + 25
    
    # Display championship standings
    standings = pd.DataFrame.from_dict(championship_points, orient='index', 
                                     columns=['Points']).sort_values('Points', ascending=False)
    print("\nPredicted Championship Standings:")
    print(standings)
    
    return season_df

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('data/2024'):
        os.makedirs('data/2024')
    
    # Run simulation
    simulated_results = simulate_2024_season()
    
    # Display detailed race-by-race results
    print("\nDetailed Race Results:")
    print("=====================")
    for _, race in simulated_results.iterrows():
        print(f"\n{race['raceName']}:")
        print(f"Winner: {race['winner']} (P={race['winProbability']:.2f})")
        print(f"Podium: {', '.join(race['topThree'])}")