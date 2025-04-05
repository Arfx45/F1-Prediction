import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import joblib

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

def simulate_2024_season():
    # Load model and data
    model, scaler = load_rnn_model()
    
    # Load 2024-specific data
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    drivers_2024 = pd.read_csv('data/2024/drivers2024.csv')
    
    # Load historical data
    drivers_df = pd.read_csv('data/drivers.csv')
    circuits_df = pd.read_csv('data/circuits.csv')
    driver_elo_df = pd.read_csv('data/driver_elo.csv')

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

    season_results = []
    season_points = {driver: 0 for driver in current_drivers}
    
    for _, race in races_2024.iterrows():
        # Assign random grid positions for simulation
        grid_positions = np.random.permutation(len(current_drivers)) + 1
        
        # Ensure circuit altitude is included
        circuit_alt = circuits_df.loc[circuits_df['circuitId'] == race['circuitId'], 'alt']
        altitude = circuit_alt.values[0] if not circuit_alt.empty else 0

        race_data = []
        
        for driver, grid in zip(current_drivers, grid_positions):
            # Retrieve driver Elo rating (fallback to mean if missing)
            driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == driver, 'Elo Rating']
            elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
            
            race_data.append({
                'driver_elo': elo_rating,
                'grid_position': grid,
                'circuit_altitude': altitude,
                'qualifying_time': 0  # Placeholder for missing qualifying data
            })
        
        # Convert race data to DataFrame and apply scaling
        race_df = pd.DataFrame(race_data)
        race_scaled = scaler.transform(race_df)

        # Reshape for RNN input
        race_scaled = race_scaled.reshape((race_scaled.shape[0], 1, race_scaled.shape[1]))

        # Predict win probabilities
        win_probabilities = model.predict(race_scaled).flatten()

        # Store race predictions
        race_predictions = pd.DataFrame({
            'Driver': current_drivers,
            'Win Probability': win_probabilities
        }).sort_values('Win Probability', ascending=False)
        
        # Compare with actual results
        comparison = compare_with_actual_results(race_predictions, race['raceId'])
        
        # Calculate points for this race
        for pos, driver in enumerate(race_predictions['Driver'], 1):
            points = calculate_points(pos)
            # Add sprint race points if applicable
            if race['name'] in ['Azerbaijan Grand Prix', 'Austrian Grand Prix', 
                                'Belgian Grand Prix', 'Qatar Grand Prix', 
                                'United States Grand Prix', 'São Paulo Grand Prix']:
                if pos <= 8:  # Sprint race points for top 8
                    sprint_points = [8, 7, 6, 5, 4, 3, 2, 1][pos-1]
                    points += sprint_points
            season_points[driver] += points
        
        race_result = {
            'raceId': race['raceId'],
            'raceName': race['name'],
            'date': race['date'],
            'predictions': race_predictions,
            'actual_comparison': comparison,
            'race_points': race_predictions.copy().assign(Points=lambda x: [calculate_points(i) for i in range(1, len(x) + 1)])
        }
        season_results.append(race_result)
    
    # Add season points summary
    season_summary = pd.DataFrame({
        'Driver': list(season_points.keys()),
        'Total Points': list(season_points.values())
    }).sort_values('Total Points', ascending=False)
    
    return season_results, season_summary

if __name__ == "__main__":
    results, points_summary = simulate_2024_season()
    
    print("\n2024 F1 Season Predictions vs Actual Results")
    print("===========================================")
    
    for race in results:
        print(f"\n{race['raceName']} ({race['date']})")
        print("-" * 50)
        print("\nTop 3 Predicted Finishers:")
        print(race['predictions'].head(3))
        print("\nPredicted Points for this Race:")
        print(race['race_points'][['Driver', 'Points']].head(10))
        print("\nComparison with Actual Result:")
        print(race['actual_comparison'])
        print("\n")
    
    print("\n2024 Championship Standings Prediction")
    print("=====================================")
    print(points_summary)
    
    # Print top 3 in championship
    print("\nPredicted Championship Podium:")
    print("-----------------------------")
    for i, (driver, points) in enumerate(zip(points_summary['Driver'].head(3), 
                                           points_summary['Total Points'].head(3)), 1):
        print(f"{i}. {driver}: {points} points")
