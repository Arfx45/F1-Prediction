import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel, RFECV
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

def create_features(row):
    # Enhanced feature engineering with proper missing value handling
    driver_elo = driver_elo_df[driver_elo_df['Driver Name'] == row['Driver Name']]['Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    circuit_alt = circuits_df[circuits_df['circuitId'] == row['circuitId']]['alt'].values[0]
    
    # Convert grid position
    grid = float(row['grid']) if row['grid'] != '\\N' else -1.0
    
    # Convert qualifying time
    q_time = float(row['milliseconds']) if row['milliseconds'] != '\\N' else 0.0
    
    # Convert laps and rank
    laps = float(row['laps']) if 'laps' in row and row['laps'] != '\\N' else 0.0
    rank = float(row['rank']) if 'rank' in row and row['rank'] != '\\N' else -1.0
    
    return pd.Series({
        'driver_elo': float(elo_rating),
        'grid_position': grid,
        'circuit_altitude': float(circuit_alt),
        'qualifying_time': q_time,
        'year': float(row['year']),
        'laps': laps,
        'fastest_lap_rank': rank
    })

# Prepare features and target
print("Preparing features...")
X = results_df.apply(create_features, axis=1)
y = (results_df['position'].astype(str).str.replace('\\N', '-1').astype(float) == 1).astype(int)

# Convert to numpy arrays and ensure float type
X = X.astype(float).to_numpy()
y = y.astype(int).to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial model for feature selection
print("Performing feature selection...")
selection_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Feature selection using SelectFromModel
selector = SelectFromModel(selection_model, prefit=False, threshold='median')
selector.fit(X_train_scaled, y_train)

# Get selected features
selected_features = [f"feature_{i}" for i in range(X_train_scaled.shape[1]) if selector.get_support()[i]]
print("\nSelected Features:")
print(selected_features)

# Transform data using selected features
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Train final model with selected features
print("\nTraining XGBoost model with selected features...")
final_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(final_model, X_train_selected, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train final model
final_model.fit(X_train_selected, y_train)

# Make predictions
y_pred = final_model.predict(X_test_selected)
y_prob = final_model.predict_proba(X_test_selected)

# Print model performance
print("\nXGBoost Model Performance with Selected Features")
print("=============================================")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance of selected features
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Selected Features):")
print(feature_importance)

# Save model, scaler, and selector
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

model_path = os.path.join('__pycache__', 'xgboost_features_F12024.joblib')
scaler_path = os.path.join('__pycache__', 'xgboost_features_scaler.joblib')
selector_path = os.path.join('__pycache__', 'xgboost_features_selector.joblib')

joblib.dump(final_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(selector, selector_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Feature selector saved to: {selector_path}")