import pandas as pd
import numpy as np
import joblib
import os

def load_xgboost_model():
    """Load the trained XGBoost model and scaler"""
    model_path = os.path.join('__pycache__', 'xgboostF12024.joblib')
    scaler_path = os.path.join('__pycache__', 'xgboost_scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Please train the model first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def calculate_points(position):
    """Calculate points based on finishing position"""
    points_system = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    return points_system.get(position, 0)

def simulate_2024_season():
    # Load model and data
    model, scaler = load_xgboost_model()
    
    # Load 2024 data
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    circuits_df = pd.read_csv('data/circuits.csv')
    driver_elo_df = pd.read_csv('data/driver_elo.csv')
    
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
    
    season_points = {driver: 0 for driver in current_drivers}
    season_results = []
    
    for _, race in races_2024.iterrows():
        # Simulate qualifying (random grid positions for simplicity)
        grid_positions = np.random.permutation(20) + 1
        
        # Prepare race data
        race_data = []
        for driver, grid in zip(current_drivers, grid_positions):
            # Get driver Elo rating
            driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver]['Elo Rating']
            elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
            
            # Get circuit altitude
            circuit_alt = circuits_df[circuits_df['circuitId'] == race['circuitId']]['alt'].values[0]
            
            race_data.append([
                elo_rating,
                grid,
                circuit_alt,
                0  # qualifying time placeholder
            ])
        
        # Scale features
        X_race = scaler.transform(np.array(race_data))
        
        # Predict probabilities
        win_probabilities = model.predict_proba(X_race)[:, 1]
        
        # Store predictions
        race_predictions = pd.DataFrame({
            'Driver': current_drivers,
            'Grid': grid_positions,
            'Win Probability': win_probabilities
        }).sort_values('Win Probability', ascending=False)
        
        # Compare with actual results if available
        actual_result = "Not yet raced"
        if race['date'] < pd.Timestamp.now().strftime('%Y-%m-%d'):
            actual_race = results_2024[results_2024['raceId'] == race['raceId']]
            if not actual_race.empty:
                actual_winner = drivers_df[
                    drivers_df['driverId'] == actual_race[actual_race['position'] == '1']['driverId'].iloc[0]
                ]
                actual_winner_name = f"{actual_winner['forename'].iloc[0]} {actual_winner['surname'].iloc[0]}"
                actual_result = f"Winner: {actual_winner_name}"
        
        # Add sprint points if applicable
        sprint_races = ['Azerbaijan Grand Prix', 'Austrian Grand Prix', 'Belgian Grand Prix', 
                       'Qatar Grand Prix', 'United States Grand Prix', 'São Paulo Grand Prix']
        
        # Store race results
        race_result = {
            'raceName': race['name'],
            'date': race['date'],
            'predictions': race_predictions,
            'actual_result': actual_result,
            'sprint_race': race['name'] in sprint_races
        }
        season_results.append(race_result)
        
        # Update season points
        for pos, driver in enumerate(race_predictions['Driver'], 1):
            points = calculate_points(pos)
            if race_result['sprint_race'] and pos <= 8:
                points += [8,7,6,5,4,3,2,1][pos-1]
            season_points[driver] += points
    
    return season_results, season_points

if __name__ == "__main__":
    results, points = simulate_2024_season()
    
    print("\n2024 F1 Season Predictions")
    print("=========================")
    
    for race in results:
        print(f"\n{race['raceName']} ({race['date']})")
        print("-" * 50)
        print("\nTop 5 Predicted Finishers:")
        print(race['predictions'][['Driver', 'Win Probability']].head())
        print(f"\nActual Result: {race['actual_result']}")
        if race['sprint_race']:
            print("* Sprint Race Weekend")
        print("\n")
    
    print("\nPredicted Championship Standings")
    print("==============================")
    standings = pd.DataFrame({
        'Driver': list(points.keys()),
        'Points': list(points.values())
    }).sort_values('Points', ascending=False)
    
    print(standings)