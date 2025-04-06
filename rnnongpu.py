import os
# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable OneDNN optimizations

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, ProgbarLogger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import datetime

# GPU Configuration
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configure GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Use only GPU 0 (RTX 3090)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"GPU detected: {gpus[0].name}")
        print("Memory growth enabled")
        
        # Set mixed precision policy
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected. Running on CPU")

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

# Set up TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

# Progress bar callback
progbar = ProgbarLogger(count_mode='steps')

# Build RNN model with GPU optimization
with tf.device('/GPU:0'):
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, 
             dtype='float16'),  # Specify dtype for mixed precision
        LSTM(64, activation='relu', return_sequences=True, 
             dtype='float16'),
        LSTM(32, activation='relu', 
             dtype='float16'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary before training
    model.summary()
    
    # Train model with callbacks and larger batch size for GPU
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,  # Increased for GPU
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[
            tensorboard_callback,
            progbar,
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/model_{epoch:02d}.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
    )

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nRNN Model Performance (Including 2023)")
print("=====================================")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

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

print("\nTo monitor training in TensorBoard, run this command in a new terminal:")
print(f"tensorboard --logdir {log_dir}")


