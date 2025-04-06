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
    model_path = os.path.join('__pycache__', 'rnnF12024.keras')
    scaler_path = os.path.join('__pycache__', 'rnn_scaler.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("RNN model not found. Please train the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Please train the model first.")
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def compare_with_actual_results(predictions, race_id):
    """Compare predictions with actual 2024 results"""
    try:
        actual_results = pd.read_csv('data/2024/results2024.csv')
        actual_race = actual_results[actual_results['raceId'] == race_id]
        
        if actual_race.empty:
            return "Race results not yet available"
            
        actual_winner = actual_race[actual_race['position'] == '1']['driverId'].iloc[0]
        actual_winner = drivers_df[
            drivers_df['driverId'] == actual_winner
        ]['forename'].iloc[0] + ' ' + drivers_df[
            drivers_df['driverId'] == actual_winner
        ]['surname'].iloc[0]
        
        comparison = {
            'Predicted Winner': predictions.iloc[0]['Driver'],
            'Actual Winner': actual_winner,
            'Prediction Accurate': predictions.iloc[0]['Driver'] == actual_winner,
            'Predicted Probability': f"{predictions.iloc[0]['Win Probability']:.2%}"
        }
        
        return pd.DataFrame([comparison])
    except FileNotFoundError:
        return "2024 results file not found"
    except Exception as e:
        return f"Error comparing results: {str(e)}"

def calculate_points(position):
    """Calculate points based on finishing position"""
    points_system = {
        1: 25,  # Win
        2: 18,  # Second
        3: 15,  # Third
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1
    }
    return points_system.get(position, 0)

def prepare_race_sequence(driver, historical_data, race_data):
    """Prepare enhanced sequential data for RNN prediction"""
    driver_data = historical_data[historical_data['Driver'] == driver]
    recent_races = driver_data.tail(5)  # Increased lookback to 5 races
    
    # Get driver Elo rating
    driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == driver, 'Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    # Get circuit altitude
    circuit_alt = circuits_df.loc[circuits_df['circuitId'] == race_data['circuitId'], 'alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0
    
    # Calculate advanced metrics with proper type handling
    recent_performance = {
        'avg_points': float(recent_races['points'].mean()) if not recent_races.empty else 0.0,
        'avg_position': float(recent_races['position'].mean()) if not recent_races.empty else 0.0,
        'dnf_rate': float(recent_races['position'].isna().mean()) if not recent_races.empty else 0.0,
        'avg_position_gain': float((recent_races['grid'] - recent_races['position']).mean()) if not recent_races.empty else 0.0,
        'win_rate': float((recent_races['position'] == 1).mean()) if not recent_races.empty else 0.0
    }
    
    return pd.Series({
        'driver_elo': float(elo_rating),
        'grid_position': -1,  # Will be set during simulation
        'circuit_altitude': float(altitude),
        'qualifying_time': 0.0,  # Placeholder for qualifying
        'recent_points': recent_performance['avg_points'],
        'recent_position': recent_performance['avg_position'],
        'dnf_rate': recent_performance['dnf_rate'],
        'position_gain': recent_performance['avg_position_gain'],
        'win_rate': recent_performance['win_rate']
    })

def load_historical_results():
    """Load and prepare historical race results"""
    historical_results = pd.read_csv('data/results.csv')
    
    # Convert string positions to numeric
    historical_results['position'] = pd.to_numeric(historical_results['position'], errors='coerce')
    historical_results['grid'] = pd.to_numeric(historical_results['grid'], errors='coerce')
    historical_results['points'] = pd.to_numeric(historical_results['points'], errors='coerce')
    
    # Merge with driver info
    historical_results = historical_results.merge(
        drivers_df[['driverId', 'forename', 'surname']], 
        on='driverId'
    )
    historical_results['Driver'] = historical_results['forename'] + ' ' + historical_results['surname']
    
    return historical_results

def simulate_rnn_2024():
    """Simulate 2024 season using enhanced RNN model"""
    model, scaler = load_rnn_model()
    
    # Load required data
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    historical_results = load_historical_results()
    
    # Calculate points for historical results
    historical_results['points'] = historical_results['position'].apply(calculate_points)
    
    season_predictions = []
    
    for _, race in races_2024.iterrows():
        print(f"\nProcessing {race['name']}...")
        race_data = []
        
        # Simulate qualifying with random grid positions
        grid_positions = np.random.permutation(len(current_drivers)) + 1
        
        for driver, grid in zip(current_drivers, grid_positions):
            # Prepare enhanced features
            features = prepare_race_sequence(driver, historical_results, race)
            features['grid_position'] = grid
            
            # Scale features
            features_scaled = scaler.transform(pd.DataFrame([features]))
            
            # Reshape for RNN (samples, timesteps, features)
            features_reshaped = features_scaled.reshape(1, 1, -1)
            
            # Predict
            win_prob = model.predict(features_reshaped, verbose=0)[0][0]
            race_data.append({
                'Driver': driver,
                'Win Probability': win_prob,
                'Grid': grid
            })
        
        # Sort by win probability
        race_results = pd.DataFrame(race_data).sort_values('Win Probability', ascending=False)
        
        # Compare with actual results if available
        actual_winner = "Not yet raced"
        prediction_correct = False
        
        if datetime.strptime(race['date'], '%Y-%m-%d').date() < datetime.now().date():
            race_result = results_2024[results_2024['raceId'] == race['raceId']]
            if not race_result.empty:
                winner_id = race_result[race_result['position'] == '1']['driverId'].iloc[0]
                winner_name = f"{drivers_df[drivers_df['driverId'] == winner_id]['forename'].iloc[0]} {drivers_df[drivers_df['driverId'] == winner_id]['surname'].iloc[0]}"
                actual_winner = winner_name
                prediction_correct = (winner_name == race_results.iloc[0]['Driver'])
        
        # Store race prediction
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
    os.makedirs('data/predicted_models', exist_ok=True)
    output_path = 'data/predicted_models/predicted_results2024RNN.csv'
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to: {output_path}")
    return predictions_df

if __name__ == "__main__":
    predictions = simulate_rnn_2024()
    
    print("\n2024 F1 Season Predictions")
    print("===========================")
    
    for race in predictions.to_dict(orient='records'):
        print(f"\n{race['raceName']} ({race['date']})")
        print("-" * 50)
        print(f"Predicted Winner: {race['predicted_winner']}")
        print(f"Win Probability: {race['win_probability']:.2%}")
        print(f"Top 3 Predicted Finishers: {', '.join(race['top_3'])}")
        print(f"Actual Winner: {race['actual_winner']}")
        print(f"Prediction Correct: {race['prediction_correct']}")
        print("\n")
