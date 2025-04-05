import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    # Get driver Elo rating with fallback to mean
    driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == row['Driver Name'], 'Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()

    # Get circuit altitude with fallback to 0
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

# Reshape for RNN input (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build RNN model
model = Sequential([
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile & train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {test_acc:.4f}")

# Save model
model.save('rnn_model.keras')


