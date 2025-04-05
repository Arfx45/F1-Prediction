import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
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

# Apply Recursive Feature Elimination (RFE)
base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_selector = RFE(base_rf, n_features_to_select=2)  # Select top 2 features
rfe_selector.fit(X_train, y_train)

# Get selected features
selected_features = X.columns[rfe_selector.support_]
print("\nSelected Features via RFE:")
print(selected_features)

# Transform the training and test sets with selected features
X_train_selected = rfe_selector.transform(X_train)
X_test_selected = rfe_selector.transform(X_test)

# Train the Random Forest model with selected features
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_selected)

# Print model performance after feature selection
print("\nRandom Forest Model Performance with Feature Selection")
print("=======================================================")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model with appropriate name
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

model_path = os.path.join('__pycache__', 'randforestF12024_with_RFE.joblib')
joblib.dump(rf_model, model_path)
print(f"\nModel saved to: {model_path}")

# Feature importance for selected features
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
})
print("\nFeature Importance (Selected Features):")
print(feature_importance.sort_values('importance', ascending=False))
