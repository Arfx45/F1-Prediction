import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load datasets
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')
circuits_df = pd.read_csv('data/circuits.csv')
driver_elo_df = pd.read_csv('data/driver_elo.csv')

# Filter data to include only up to 2023
races_df = races_df[races_df['year'] < 2024]
print(f"Training on data from years: {races_df['year'].unique()}")

# Merge datasets
results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')
results_df = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
results_df['Driver Name'] = results_df['forename'] + ' ' + results_df['surname']

# Feature engineering
def create_features(row):
    # Get driver Elo rating
    driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == row['Driver Name']]['Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    # Get circuit altitude
    circuit_alt = circuits_df[circuits_df['circuitId'] == row['circuitId']]['alt'].values[0]
    
    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': row['grid'] if row['grid'] != '\\N' else -1,
        'circuit_altitude': circuit_alt,
        'qualifying_time': row['milliseconds'] if row['milliseconds'] != '\\N' else 0
    })

# Prepare features and target
print("Preparing features...")
X = results_df.apply(create_features, axis=1)
y = (results_df['position'].astype(str).str.replace('\\N', '-1').astype(float) == 1).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)
y_prob = xgb_model.predict_proba(X_test_scaled)

# Print model performance
print("\nXGBoost Model Performance")
print("========================")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model and scaler
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

model_path = os.path.join('__pycache__', 'xgboostF12024.joblib')
scaler_path = os.path.join('__pycache__', 'xgboost_scaler.joblib')

joblib.dump(xgb_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")