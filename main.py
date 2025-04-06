import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import os

def load_all_models():
    """Load all trained models including feature selection variants"""
    models = {}
    
    try:
        # Base models
        models['Random Forest'] = joblib.load('__pycache__/randforestF12024.joblib')
        models['Random Forest FS'] = joblib.load('__pycache__/randforestFS_F12024.joblib')
        
        models['XGBoost'] = joblib.load('__pycache__/xgboostF12024.joblib')
        models['XGBoost FS'] = joblib.load('__pycache__/xgboostFS_F12024.joblib')
        
        models['SVM'] = joblib.load('__pycache__/svmF12024.joblib')
        models['SVM FS'] = joblib.load('__pycache__/svmFS_F12024.joblib')
        
        models['Logistic'] = joblib.load('__pycache__/logisticF12024.joblib')
        models['Logistic FS'] = joblib.load('__pycache__/logisticFS_F12024.joblib')
        
        models['RNN'] = tf.keras.models.load_model('__pycache__/rnnF12024.keras')
        models['RNN FS'] = tf.keras.models.load_model('__pycache__/rnnFS_F12024.keras')
        
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def calculate_points(position):
    """Calculate F1 points based on finishing position"""
    points_system = {
        '1': 25, '2': 18, '3': 15, '4': 12, '5': 10,
        '6': 8, '7': 6, '8': 4, '9': 2, '10': 1
    }
    return points_system.get(str(position), 0)

def visualize_model_comparisons():
    """Create visualizations comparing all model predictions with actual 2024 results"""
    
    # Load actual 2024 results
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    
    # Calculate actual standings and points
    actual_results = results_2024.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId')
    actual_results['Driver'] = actual_results['forename'] + ' ' + actual_results['surname']
    actual_results['Points'] = actual_results['position'].apply(calculate_points)
    actual_standings = actual_results.groupby('Driver')['Points'].sum().sort_values(ascending=False)
    
    # Load only base model predictions with correct paths
    prediction_files = {
        'Random Forest': 'data/predicted_models/predicted_results2024RF.csv',
        'XGBoost': 'data/predicted_models/predicted_results2024XGB.csv',
        'SVM': 'data/predicted_models/predicted_results2024SVM.csv',
        'Logistic': 'data/predicted_models/predicted_results2024Logistic.csv',
        'RNN': 'data/predicted_models/predicted_results2024RNN.csv'
    }
    
    # Store accuracies and predictions
    accuracies = {}
    standings_mae = {}
    
    # Create comparison visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. Race Winner Prediction Accuracy
    plt.subplot(2, 2, 1)
    models = []
    accs = []
    
    for model, pred_file in prediction_files.items():
        try:
            pred_df = pd.read_csv(pred_file)
            
            # Handle cases where prediction_correct column doesn't exist
            if 'prediction_correct' not in pred_df.columns:
                # Calculate prediction_correct by comparing predicted_winner with actual_winner
                pred_df['prediction_correct'] = pred_df['predicted_winner'] == pred_df['actual_winner']
            
            acc = pred_df['prediction_correct'].mean()
            accuracies[model] = acc
            models.append(model)
            accs.append(acc)
        except Exception as e:
            print(f"Skipping {model} - Could not process predictions: {str(e)}")
            continue
    
    # Plot bar chart
    bars = plt.bar(range(len(models)), accs, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
    plt.title('Race Winner Prediction Accuracy by Model')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(models)), models, rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    # 2. Driver Standings Comparison
    plt.subplot(2, 2, 2)
    top_10_actual = actual_standings.head(10)
    plt.plot(range(len(top_10_actual)), top_10_actual.values, 'k-', label='Actual', linewidth=2)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for model, color in zip(models, colors):
        try:
            pred_df = pd.read_csv(prediction_files[model])
            pred_standings = pred_df.groupby('predicted_winner')['win_probability'].count().sort_values(ascending=False)
            plt.plot(range(len(pred_standings.head(10))), pred_standings.head(10).values, '--', 
                    label=model, color=color, alpha=0.7)
            
            standings_mae[model] = np.mean(np.abs(
                pred_standings.head(10).values - top_10_actual.head(10).values
            ))
        except Exception:
            continue
    
    plt.title('Top 10 Drivers Points/Wins Comparison')
    plt.xlabel('Driver Ranking')
    plt.ylabel('Points/Predicted Wins')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    os.makedirs('model_comparison', exist_ok=True)
    plt.savefig('model_comparison/model_comparison_2024.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nModel Performance Summary")
    print("=======================")
    print("\nRace Winner Prediction Accuracy:")
    for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:15s}: {acc:.2%}")
    
    print("\nStandings Prediction Error (MAE):")
    for model, mae in sorted(standings_mae.items(), key=lambda x: x[1]):
        print(f"{model:15s}: {mae:.2f}")

def main():
    print("F1 2024 Season Model Comparison")
    print("===============================")
    visualize_model_comparisons()
    print("\nVisualizations saved to model_comparison/model_comparison_2024.png")

if __name__ == "__main__":
    main()