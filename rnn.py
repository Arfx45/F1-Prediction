import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# Load datasets
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
drivers_df = pd.read_csv('data/drivers.csv')
circuits_df = pd.read_csv('data/circuits.csv')
driver_elo_df = pd.read_csv('data/driver_elo.csv')

# Merge and filter data for recent seasons (excluding 2024)
races_df = races_df[races_df['year'] < 2024]
print(f"Training on data from years: {races_df['year'].unique()}")
results_df = results_df.merge(races_df[['raceId', 'year', 'circuitId']], on='raceId')
results_df = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')

# Add driver name for easier lookup
results_df['Driver Name'] = results_df['forename'] + ' ' + results_df['surname']

# Replace missing values or non-numeric values
results_df['grid'] = results_df['grid'].replace('\\N', '-1').astype(float)
results_df['milliseconds'] = results_df['milliseconds'].replace('\\N', '0').astype(float)
results_df['position'] = results_df['position'].replace('\\N', '-1').astype(float)

# Add performance metrics calculation
def calculate_performance_metrics(df):
    df = df.copy()
    df['points'] = df['position'].apply(lambda x: {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }.get(x, 0) if x > 0 else 0)
    
    df['position_gain'] = df['grid'] - df['position']
    df['dnf'] = (df['position'] == -1).astype(int)
    return df

# Enhanced feature engineering
def create_features(row, historical_data):
    driver_data = historical_data[historical_data['Driver Name'] == row['Driver Name']]
    recent_races = driver_data[driver_data['raceId'] < row['raceId']].tail(5)
    
    # Get base features
    driver_elo = driver_elo_df.loc[driver_elo_df['Driver Name'] == row['Driver Name'], 'Elo Rating']
    elo_rating = driver_elo.values[0] if not driver_elo.empty else driver_elo_df['Elo Rating'].mean()
    
    circuit_alt = circuits_df.loc[circuits_df['circuitId'] == row['circuitId'], 'alt']
    altitude = circuit_alt.values[0] if not circuit_alt.empty else 0
    
    # Calculate advanced metrics
    recent_performance = {
        'avg_points': recent_races['points'].mean() if not recent_races.empty else 0,
        'avg_position': recent_races['position'].mean() if not recent_races.empty else 0,
        'dnf_rate': recent_races['dnf'].mean() if not recent_races.empty else 0,
        'avg_position_gain': recent_races['position_gain'].mean() if not recent_races.empty else 0,
        'win_rate': (recent_races['position'] == 1).mean() if not recent_races.empty else 0
    }
    
    return pd.Series({
        'driver_elo': elo_rating,
        'grid_position': row['grid'],
        'circuit_altitude': altitude,
        'qualifying_time': row['milliseconds'],
        'recent_points': recent_performance['avg_points'],
        'recent_position': recent_performance['avg_position'],
        'dnf_rate': recent_performance['dnf_rate'],
        'position_gain': recent_performance['avg_position_gain'],
        'win_rate': recent_performance['win_rate']
    })

# Process data
results_df = calculate_performance_metrics(results_df)
X = results_df.apply(lambda row: create_features(row, results_df), axis=1)
y = (results_df['position'] == 1).astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for RNN input (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build improved RNN model
model = Sequential([
    LSTM(256, activation='relu', return_sequences=True, input_shape=(1, X_scaled.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(128, activation='relu', return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    
    LSTM(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

# Compile with better optimizer settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Add callbacks for better training
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001)
]

# Train with more epochs and callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
metrics = model.evaluate(X_test, y_test)
print("\nRNN Model Performance (Including 2023)")
print("=====================================")
print(f"Test Loss: {metrics[0]:.4f}")
print(f"Test Accuracy: {metrics[1]:.4f}")
print(f"Test Precision: {metrics[2]:.4f}")
print(f"Test Recall: {metrics[3]:.4f}")
print(f"F1 Score: {2 * (metrics[2] * metrics[3]) / (metrics[2] + metrics[3]):.4f}")

# Save model and scaler
if not os.path.exists('__pycache__'):
    os.makedirs('__pycache__')

# Save the model
model_path = os.path.join('__pycache__', 'rnnF12024.keras')
model.save(model_path)

# Save the scaler
scaler_path = os.path.join('__pycache__', 'rnn_scaler.joblib')
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")

# Print training history summary
print("\nTraining History:")
print("================")
for metric in history.history.keys():
    final_value = history.history[metric][-1]
    print(f"{metric}: {final_value:.4f}")


