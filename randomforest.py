import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load the necessary data
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')
circuits_df = pd.read_csv('data/circuits.csv')
driver_elo_df = pd.read_csv('data/driver_elo.csv')

# Filter to include 2023 but exclude 2024
races_df = races_df[races_df['year'] < 2024]
print(f"Training on data from years: {races_df['year'].unique()}")
results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')

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

# Create and train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print model performance with updated date range
print("\nRandom Forest Model Performance (Including 2023)")
print("=============================================")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model with appropriate name
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')
    
model_path = os.path.join('__pycache__', 'randforestF12024.joblib')
joblib.dump(rf_model, model_path)
print(f"\nModel saved to: {model_path}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

def predict_race_winner(grid_positions, driver_names, circuit_id, qualifying_times=None):
    race_data = []
    
    # Load results data for qualifying reference
    results_df = pd.read_csv('data/results.csv')
    
    for i, (driver, grid) in enumerate(zip(driver_names, grid_positions)):
        # Get driver Elo rating with fallback to mean
        driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == driver]['Elo Rating']
        elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
        
        # Get circuit altitude
        circuit_alt = circuits_df[circuits_df['circuitId'] == circuit_id]['alt'].values[0]
        
        # Get driver's qualifying time from results data
        driver_id = drivers_df[
            (drivers_df['forename'] + ' ' + drivers_df['surname']) == driver
        ]['driverId'].values[0]
        
        # Get qualifying time from results where grid position matches
        qualifying_result = results_df[
            (results_df['driverId'] == driver_id) & 
            (results_df['grid'] == grid) &
            (results_df['circuitId'] == circuit_id)
        ]['milliseconds'].min()
        
        # Use qualifying result if available, otherwise use provided time or 0
        q_time = qualifying_result if pd.notna(qualifying_result) else (
            qualifying_times[i] if qualifying_times is not None else 0
        )
        
        race_data.append({
            'driver_elo': elo_rating,
            'grid_position': grid,
            'circuit_altitude': circuit_alt,
            'qualifying_time': q_time
        })
    
    # Convert to DataFrame
    race_df = pd.DataFrame(race_data)
    
    # Make predictions
    win_probabilities = rf_model.predict_proba(race_df)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Driver': driver_names,
        'Grid': grid_positions,
        'Win Probability': win_probabilities
    }).sort_values('Win Probability', ascending=False)
    
    return results

# Example usage
if __name__ == "__main__":
    sample_drivers = ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc']
    sample_grid = [1, 2, 3]
    circuit_id = 1  # Example circuit ID
    
    predictions = predict_race_winner(
        grid_positions=sample_grid,
        driver_names=sample_drivers,
        circuit_id=circuit_id
    )
    
    print("\nPredicted Race Results:")
    print("=====================")
    print(predictions)