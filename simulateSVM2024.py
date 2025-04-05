import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained SVM model
svm_model = joblib.load('svm_model.pkl')

# Load 2024 race data
races_2024 = pd.read_csv('data/2024/races2024.csv')
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

# Initialize scaler
scaler = StandardScaler()

season_predictions = []

for _, race in races_2024.iterrows():
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
            'qualifying_time': 0  # Placeholder for missing qualifying data
        })
    
    # Convert race data to DataFrame and apply scaling
    race_df = pd.DataFrame(race_data)
    race_scaled = scaler.fit_transform(race_df)

    # Predict win probabilities using SVM
    win_probs = svm_model.predict_proba(race_scaled)[:, 1]

    # Store race predictions
    race_results = pd.DataFrame({'Driver': current_drivers, 'Win Probability': win_probs}).sort_values('Win Probability', ascending=False)
    
    season_predictions.append({
        'raceId': race['raceId'],
        'raceName': race['name'],
        'date': race['date'],
        'predicted_winner': race_results.iloc[0]['Driver'],
        'win_probability': race_results.iloc[0]['Win Probability'],
        'top_3': race_results.head(3)['Driver'].tolist()
    })

# Convert results to DataFrame and save
predictions_df = pd.DataFrame(season_predictions)
predictions_df.to_csv('data/2024/predicted_results2024SVM.csv', index=False)

# Print race predictions
for race in season_predictions:
    print(f"\n{race['raceName']} ({race['date']}):")
    print(f"Predicted Winner: {race['predicted_winner']} (P={race['win_probability']:.2f})")
    print(f"Predicted Podium: {', '.join(race['top_3'])}")

print("\nSaved predictions to 'data/2024/predicted_results2024SVM.csv'")
