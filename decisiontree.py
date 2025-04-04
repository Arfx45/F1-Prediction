import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the necessary data
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')
circuits_df = pd.read_csv('data/circuits.csv')
driver_elo_df = pd.read_csv('data/driver_elo.csv')

# Filter out 2024 data
races_df = races_df[races_df['year'] < 2024]
results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')

# Create feature dataframe
def create_features(row, driver_elo_df, circuits_df):
    # Get driver Elo rating with fallback to mean
    driver_name = row['Driver Name']
    driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver_name]['Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    # Get circuit altitude with fallback to 0
    circuit_alt = circuits_df[circuits_df['circuitId'] == row['circuitId']]['alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0
    
    features = {
        'driver_elo': elo_rating,
        'grid_position': row['grid'] if row['grid'] != '\\N' else -1,
        'circuit_altitude': altitude,
        'qualifying_time': row['milliseconds'] if 'milliseconds' in row and row['milliseconds'] != '\\N' else 0
    }
    return pd.Series(features)

# Prepare the data
merged_data = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
merged_data['Driver Name'] = merged_data['forename'] + ' ' + merged_data['surname']

# Create features and target
X = merged_data.apply(lambda x: create_features(x, driver_elo_df, circuits_df), axis=1)
y = (merged_data['position'].astype(str).str.replace('\\N', '-1').astype(float) == 1).astype(int)

# Remove any rows with NaN values
valid_indices = ~X.isna().any(axis=1)
X = X[valid_indices]
y = y[valid_indices]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Print model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Function to predict winner for a new race
def predict_race_winner(grid_positions, driver_names, circuit_id):
    race_data = []
    for driver, grid in zip(driver_names, grid_positions):
        driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver]['Elo Rating'].values[0]
        circuit_alt = circuits_df[circuits_df['circuitId'] == circuit_id]['alt'].values[0]
        
        race_data.append({
            'driver_elo': driver_elo,
            'grid_position': grid,
            'circuit_altitude': circuit_alt,
            'qualifying_time': 0  # You can add qualifying time if available
        })
    
    race_df = pd.DataFrame(race_data)
    predictions = dt_model.predict_proba(race_df)
    win_probabilities = predictions[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'Driver': driver_names,
        'Win Probability': win_probabilities
    })
    return results.sort_values('Win Probability', ascending=False)

# Example usage:
# sample_drivers = ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc']
# sample_grid = [1, 2, 3]
# circuit_id = 1  # Example circuit ID
# predictions = predict_race_winner(sample_grid, sample_drivers, circuit_id)
# print("\nPredicted Race Winner Probabilities:")
# print(predictions)