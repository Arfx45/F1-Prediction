import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

def simulate_xgb_2024():
    """Simulate 2024 season using XGBoost model"""
    # Load model and data
    model = joblib.load('__pycache__/xgboostF12024.joblib')
    scaler = joblib.load('__pycache__/xgboost_scaler.joblib')
    
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    circuits_df = pd.read_csv('data/circuits.csv')
    driver_elo_df = pd.read_csv('data/driver_elo.csv')

    # Current F1 drivers
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

    for _, race in races_2024.iterrows():
        print(f"\nProcessing {race['name']}...")
        
        # Simulate qualifying
        grid_positions = np.random.permutation(len(current_drivers)) + 1
        circuit_alt = circuits_df.loc[circuits_df['circuitId'] == race['circuitId'], 'alt']
        altitude = circuit_alt.values[0] if not circuit_alt.empty else 0

        race_data = []
        
        for driver, grid in zip(current_drivers, grid_positions):
            driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == driver, 'Elo Rating']
            elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
            
            race_data.append({
                'driver_elo': elo_rating,
                'grid_position': grid,
                'circuit_altitude': altitude,
                'qualifying_time': 0  # Placeholder for missing qualifying data
            })
        
        race_df = pd.DataFrame(race_data)
        expected_columns = ['driver_elo', 'grid_position', 'circuit_altitude', 'qualifying_time']
        race_df = race_df[expected_columns]
        
        race_scaled = scaler.transform(race_df)

        # Predict win probabilities
        win_probs = model.predict_proba(race_scaled)[:, 1]
        race_results = pd.DataFrame({
            'Driver': current_drivers,
            'Win Probability': win_probs
        }).sort_values('Win Probability', ascending=False)
        
        # Compare with actual results if available
        actual_winner = "Not yet raced"
        prediction_correct = None
        
        if datetime.strptime(race['date'], '%Y-%m-%d').date() < datetime.now().date():
            race_result = results_2024[results_2024['raceId'] == race['raceId']]
            if not race_result.empty:
                winner_id = race_result[race_result['position'] == '1']['driverId'].iloc[0]
                winner_name = f"{drivers_df[drivers_df['driverId'] == winner_id]['forename'].iloc[0]} {drivers_df[drivers_df['driverId'] == winner_id]['surname'].iloc[0]}"
                actual_winner = winner_name
                prediction_correct = (winner_name == race_results.iloc[0]['Driver'])
        
        season_predictions.append({
            'raceId': race['raceId'],
            'raceName': race['name'],
            'date': race['date'],
            'predicted_winner': race_results.iloc[0]['Driver'],
            'win_probability': race_results.iloc[0]['Win Probability'],
            'top_3': race_results.head(3)['Driver'].tolist(),
            'actual_winner': actual_winner,
            'prediction_correct': prediction_correct
        })

    # Create predictions DataFrame and save to CSV
    predictions_df = pd.DataFrame(season_predictions)
    
    # Ensure directory exists
    os.makedirs('data/predicted_models', exist_ok=True)
    
    # Save predictions
    output_path = 'data/predicted_models/predicted_results2024XGB.csv'
    predictions_df.to_csv(output_path, index=False)
    
    # Print results summary
    correct_predictions = sum(p for p in predictions_df['prediction_correct'] if p is not None)
    total_predictions = sum(1 for p in predictions_df['prediction_correct'] if p is not None)
    
    print("\nXGBoost 2024 Season Predictions")
    print("==============================")
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nAccuracy on completed races: {accuracy:.2%}")
    
    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    simulate_xgb_2024()