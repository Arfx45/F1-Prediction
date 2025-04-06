import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')
results_df = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
results_df['Driver Name'] = results_df['forename'] + ' ' + results_df['surname']

# Handle missing values
results_df['grid'] = results_df['grid'].replace('\\N', '-1').astype(float)
results_df['milliseconds'] = results_df['milliseconds'].replace('\\N', '0').astype(float)
results_df['position'] = results_df['position'].replace('\\N', '-1').astype(float)

def create_features(row):
    driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == row['Driver Name'], 'Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    circuit_alt = circuits_df.loc[circuits_df['circuitId'] == row['circuitId'], 'alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0
    
    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': row['grid'],
        'circuit_altitude': altitude,
        'qualifying_time': row['milliseconds'],
        'year': row['year']
    })

# Prepare features and target
print("Preparing features...")
X = results_df.apply(create_features, axis=1)
y = (results_df['position'] == 1).astype(int)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
print("Training SVM model...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = svm_model.predict(X_test_scaled)
print(f"\nSVM Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

model_path = os.path.join('__pycache__', 'svmF12024.joblib')
scaler_path = os.path.join('__pycache__', 'svm_scaler.joblib')

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
