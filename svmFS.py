import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
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

# Merge and filter data for recent seasons
races_df = races_df[races_df['year'].between(2018, 2023)]
results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')
results_df = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')

# Add driver name for easier lookup
results_df['Driver Name'] = results_df['forename'] + ' ' + results_df['surname']

# Replace missing values or non-numeric values
results_df['grid'] = results_df['grid'].replace('\\N', '-1').astype(float)
results_df['milliseconds'] = results_df['milliseconds'].replace('\\N', '0').astype(float)
results_df['position'] = results_df['position'].replace('\\N', '-1').astype(float)

# Feature engineering function
def create_features(row):
    driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == row['Driver Name'], 'Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()

    circuit_alt = circuits_df.loc[circuits_df['circuitId'] == row['circuitId'], 'alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0

    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': row['grid'],
        'circuit_altitude': altitude,
        'qualifying_time': row['milliseconds']
    })

# Apply feature creation
X = results_df.apply(create_features, axis=1)
y = (results_df['position'] == 1).astype(int)  # Target: 1 if the driver won, 0 otherwise

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature selection
print("Performing feature selection...")
svm_selector = SVC(kernel='linear', random_state=42)
selector = RFE(estimator=svm_selector, n_features_to_select=3)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train SVM with selected features
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_selected, y_train)

# Evaluate and save model
y_pred = svm_model.predict(X_test_selected)
print(f"\nSVM Model with Feature Selection Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model, scaler, and selector
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

model_path = os.path.join('__pycache__', 'svmFS_F12024.joblib')
scaler_path = os.path.join('__pycache__', 'svmFS_scaler.joblib')
selector_path = os.path.join('__pycache__', 'svmFS_selector.joblib')

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(selector, selector_path)
