import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_svm_model():
    """Load the trained SVM model and scaler"""
    try:
        model_path = os.path.join('__pycache__', 'svmF12024.joblib')
        scaler_path = os.path.join('__pycache__', 'svm_scaler.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model files not found. Please train the SVM model first.")
        
        print("Loading SVM model and scaler...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def simulate_svm_2024():
    print("\nStarting 2024 F1 Season Simulation with SVM Model")
    print("=" * 50)
    
    # Load model and data
    svm_model, scaler = load_svm_model()

    # Load 2024 race data
    print("\nLoading 2024 season data...")
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
    correct_predictions = 0
    total_predictions = 0

    print("\nSimulating races...")
    for _, race in races_2024.iterrows():
        print(f"\nProcessing {race['name']}...")
        
        # Assign random grid positions for simulation
        grid_positions = np.random.permutation(len(current_drivers)) + 1
        
        # Ensure circuit altitude is included
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
                'qualifying_time': 0,  # Placeholder for missing qualifying data
                'year': 2024  # Add the year feature
            })
        
        # Convert race data to DataFrame and apply scaling
        race_df = pd.DataFrame(race_data)

        # Ensure columns match training data
        expected_columns = ['driver_elo', 'grid_position', 'circuit_altitude', 'qualifying_time', 'year']
        race_df = race_df[expected_columns]
        
        race_scaled = scaler.transform(race_df)

        # Predict win probabilities using SVM
        win_probs = svm_model.predict_proba(race_scaled)[:, 1]

        # Store race predictions
        race_results = pd.DataFrame({'Driver': current_drivers, 'Win Probability': win_probs}).sort_values('Win Probability', ascending=False)
        
        # Add actual winner comparison
        actual_winner = "Not yet raced"
        if datetime.strptime(race['date'], '%Y-%m-%d').date() < datetime.now().date():
            race_result = results_2024[results_2024['raceId'] == race['raceId']]
            if not race_result.empty:
                winner_id = race_result[race_result['position'] == '1']['driverId'].iloc[0]
                actual_winner = f"{drivers_df[drivers_df['driverId'] == winner_id]['forename'].iloc[0]} {drivers_df[drivers_df['driverId'] == winner_id]['surname'].iloc[0]}"
        
        season_predictions.append({
            'raceId': race['raceId'],
            'raceName': race['name'],
            'date': race['date'],
            'predicted_winner': race_results.iloc[0]['Driver'],
            'win_probability': race_results.iloc[0]['Win Probability'],
            'top_3': race_results.head(3)['Driver'].tolist(),
            'actual_winner': actual_winner,
            'prediction_correct': (actual_winner != "Not yet raced" and actual_winner == race_results.iloc[0]['Driver'])
        })

        # Add actual results comparison
        if datetime.strptime(race['date'], '%Y-%m-%d').date() < datetime.now().date():
            race_result = results_2024[results_2024['raceId'] == race['raceId']]
            if not race_result.empty:
                actual_winner_id = race_result[race_result['position'] == '1']['driverId'].iloc[0]
                actual_winner = f"{drivers_df[drivers_df['driverId'] == actual_winner_id]['forename'].iloc[0]} {drivers_df[drivers_df['driverId'] == actual_winner_id]['surname'].iloc[0]}"
                
                prediction_correct = (actual_winner == race_results.iloc[0]['Driver'])
                total_predictions += 1
                if prediction_correct:
                    correct_predictions += 1
                
                print(f"Predicted Winner: {race_results.iloc[0]['Driver']}")
                print(f"Actual Winner: {actual_winner}")
                print(f"Prediction {'Correct' if prediction_correct else 'Incorrect'}")
            else:
                print("Race results not yet available")
        else:
            print(f"Race not yet completed. Predicted winner: {race_results.iloc[0]['Driver']}")

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(season_predictions)
    
    # Ensure directory exists
    os.makedirs('data/predicted_models', exist_ok=True)
    
    # Save predictions
    output_path = 'data/predicted_models/predicted_results2024SVM.csv'
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    return season_predictions

if __name__ == "__main__":
    try:
        simulate_svm_2024()
    except Exception as e:
        print(f"Error during simulation: {e}")
