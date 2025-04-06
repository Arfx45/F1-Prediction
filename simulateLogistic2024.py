import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def load_logistic_model():
    """Load the trained Logistic Regression model and scaler"""
    try:
        model_path = os.path.join('__pycache__', 'logisticF12024.joblib')
        scaler_path = os.path.join('__pycache__', 'logistic_scaler.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please ensure model files exist in __pycache__ directory")
        raise

def simulate_2024_season():
    # Load model and scaler
    log_model, scaler = load_logistic_model()
    
    # Load 2024-specific data
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
        
        # Convert to DataFrame and ensure column order matches training data
        race_df = pd.DataFrame(race_data)
        expected_columns = ['driver_elo', 'grid_position', 'circuit_altitude', 'qualifying_time']
        race_df = race_df[expected_columns]
        
        race_scaled = scaler.transform(race_df)

        # Predict win probabilities
        win_probs = log_model.predict_proba(race_scaled)[:, 1]
        race_results = pd.DataFrame({
            'Driver': current_drivers,
            'Win Probability': win_probs
        }).sort_values(by='Win Probability', ascending=False)
        
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
    
    # Save predictions
    predictions_df = pd.DataFrame(season_predictions)
    predictions_df.to_csv('data/2024/predicted_results2024Logistic.csv', index=False)
    
    # Print detailed results
    print("\nLogistic Regression 2024 Season Predictions")
    print("=========================================")
    correct_predictions = 0
    total_predictions = 0
    
    for race in season_predictions:
        print(f"\n{race['raceName']} ({race['date']}):")
        print(f"Predicted Winner: {race['predicted_winner']} (P={race['win_probability']:.2f})")
        print(f"Predicted Podium: {', '.join(race['top_3'])}")
        print(f"Actual Winner: {race['actual_winner']}")
        if race['prediction_correct'] is not None:
            total_predictions += 1
            if race['prediction_correct']:
                correct_predictions += 1
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nPrediction Accuracy for completed races: {accuracy:.2%}")
    
    print("\nSaved predictions to 'data/2024/predicted_results2024Logistic.csv'")

if __name__ == "__main__":
    simulate_2024_season()
