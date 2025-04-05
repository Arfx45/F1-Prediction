import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

# Apply Recursive Feature Elimination (RFE)
base_svc = SVC(kernel='linear', class_weight={0: 1, 1: 4})  # Use linear kernel for RFE
rfe_selector = RFE(base_svc, n_features_to_select=2)  # Select top 2 features
rfe_selector.fit(X_scaled, y)

# Get selected features
selected_features = X.columns[rfe_selector.support_]
print("\nSelected Features via RFE:")
print(selected_features)

# Transform the dataset to include only selected features
X_selected = rfe_selector.transform(X_scaled)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train SVM model with selected features
svm_model = SVC(kernel='rbf', C=4.0, gamma='scale', class_weight={0: 1, 1: 4}, probability=True)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate model performance
print("\nSVM Model Performance with Feature Selection")
print("============================================")
print(f"Model Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# Save model
joblib.dump(svm_model, 'svm_model_with_RFE.pkl')
