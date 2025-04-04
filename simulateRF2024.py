import pandas as pd
import numpy as np
from randomforest import rf_model, predict_race_winner
import os

def simulate_2024_season():
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

    # Simulate each race
    for _, race in races_2024.iterrows():
        # Simulate qualifying (random grid positions for simplicity)
        grid_positions = np.random.permutation(20) + 1
        
        # Predict race outcome
        predictions = predict_race_winner(
            grid_positions=grid_positions.tolist(),
            driver_names=current_drivers,
            circuit_id=race['circuitId']
        )
        
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