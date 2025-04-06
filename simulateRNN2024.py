import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import joblib
from datetime import datetime

# Load required datasets
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

def load_rnn_model():
    """Load trained RNN model and scaler"""
    model_path = '__pycache__/rnnF12024.keras'
    scaler_path = '__pycache__/rnn_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train the model first.")
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def create_features(driver, race_data):
    """Create features for prediction with all required columns"""
    # Get driver Elo rating
    driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == driver, 'Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    # Get circuit altitude
    circuit_alt = circuits_df.loc[circuits_df['circuitId'] == race_data['circuitId'], 'alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0
    
    # Create feature set matching the trained model
    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': -1,  # Will be set during simulation
        'circuit_altitude': altitude,
        'qualifying_time': 0,  # Placeholder
        'recent_points': 0,    # Add missing features
        'recent_position': 0,
        'dnf_rate': 0,
        'position_gain': 0,
        'win_rate': 0
    })

def simulate_rnn_2024():
    """Simulate 2024 F1 season using RNN model"""
    print("Loading RNN model and data...")
    model, scaler = load_rnn_model()
    
    # Load 2024 race data
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    
    season_predictions = []
    
    for _, race in races_2024.iterrows():
        print(f"\nSimulating {race['name']}...")
        race_data = []
        
        # Simulate qualifying
        grid_positions = np.random.permutation(len(current_drivers)) + 1
        
        for driver, grid in zip(current_drivers, grid_positions):
            # Prepare features
            features = create_features(driver, race)
            features['grid_position'] = grid
            
            # Scale and reshape for RNN
            features_scaled = scaler.transform(pd.DataFrame([features]))
            features_reshaped = features_scaled.reshape(1, 1, -1)
            
            # Get win probability
            win_prob = model.predict(features_reshaped, verbose=0)[0][0]
            race_data.append({
                'Driver': driver,
                'Win Probability': win_prob,
                'Grid': grid
            })
        
        # Sort by win probability
        race_results = pd.DataFrame(race_data).sort_values('Win Probability', ascending=False)
        
        # Get actual winner if race completed
        actual_winner = "Not yet raced"
        prediction_correct = None
        
        if datetime.strptime(race['date'], '%Y-%m-%d').date() < datetime.now().date():
            race_result = results_2024[results_2024['raceId'] == race['raceId']]
            if not race_result.empty:
                winner_id = race_result[race_result['position'] == '1']['driverId'].iloc[0]
                winner_name = f"{drivers_df[drivers_df['driverId'] == winner_id]['forename'].iloc[0]} {drivers_df[drivers_df['driverId'] == winner_id]['surname'].iloc[0]}"
                actual_winner = winner_name
                prediction_correct = (winner_name == race_results.iloc[0]['Driver'])
        
        # Store prediction
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
        
        print(f"Predicted winner: {race_results.iloc[0]['Driver']} ({race_results.iloc[0]['Win Probability']:.2%})")
    
    # Save predictions
    os.makedirs('data/predicted_models', exist_ok=True)
    predictions_df = pd.DataFrame(season_predictions)
    output_path = 'data/predicted_models/predicted_results2024RNN.csv'
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to: {output_path}")
    return predictions_df

if __name__ == "__main__":
    predictions = simulate_rnn_2024()
    print("\nSimulation complete.")