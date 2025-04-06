import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load datasets
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')
circuits_df = pd.read_csv('data/circuits.csv')
driver_elo_df = pd.read_csv('data/driver_elo.csv')

# Filter for recent seasons
races_df = races_df[races_df['year'].between(2018, 2023)]
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
    
    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': row['grid'] if row['grid'] != '\\N' else -1,
        'circuit_altitude': altitude,
        'qualifying_time': row['milliseconds'] if 'milliseconds' in row and row['milliseconds'] != '\\N' else 0
    })

# Prepare data
merged_data = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
merged_data['Driver Name'] = merged_data['forename'] + ' ' + merged_data['surname']

# Create features and target
X = merged_data.apply(lambda x: create_features(x, driver_elo_df, circuits_df), axis=1)
y = (merged_data['position'].astype(str).str.replace('\\N', '-1').astype(float) == 1).astype(int)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_scaled)
print("\nRandom Forest Model Performance")
print("==============================")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

joblib.dump(rf_model, '__pycache__/randforestF12024.joblib')
joblib.dump(scaler, '__pycache__/rf_scaler.joblib')
print("\nModel and scaler saved to __pycache__ directory")