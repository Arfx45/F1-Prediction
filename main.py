import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
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

def visualize_model_accuracy():
    """Create visualization comparing all model accuracies"""
    # Load 2024 data
    results_2024 = pd.read_csv('data/2024/results2024.csv')
    races_2024 = pd.read_csv('data/2024/races2024.csv')
    
    # Store accuracies
    accuracies = {}
    
    # Calculate accuracy for each model's predictions
    prediction_files = {
        'Random Forest': 'predicted_results2024RF.csv',
        'Random Forest FS': 'predicted_results2024RF_FS.csv',
        'XGBoost': 'predicted_results2024XGB.csv',
        'XGBoost FS': 'predicted_results2024XGB_FS.csv',
        'SVM': 'predicted_results2024SVM.csv',
        'SVM FS': 'predicted_results2024SVM_FS.csv',
        'Logistic': 'predicted_results2024Logistic.csv',
        'Logistic FS': 'predicted_results2024Logistic_FS.csv',
        'RNN': 'predicted_results2024RNN.csv',
        'RNN FS': 'predicted_results2024RNN_FS.csv'
    }
    
    # Check which predictions exist
    available_models = {}
    for model_name, pred_file in prediction_files.items():
        filepath = f'data/2024/{pred_file}'
        if os.path.exists(filepath):
            try:
                pred_df = pd.read_csv(filepath)
                accuracies[model_name] = pred_df['prediction_correct'].mean()
                available_models[model_name] = pred_file
            except Exception as e:
                print(f"Error loading {model_name} predictions: {e}")
    
    if not accuracies:
        print("No prediction files found. Please run model simulations first.")
        return
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    colors = sns.color_palette('husl', n_colors=len(accuracies))
    
    # Create grouped bar plot for available models
    base_models = [k for k in accuracies.keys() if 'FS' not in k]
    fs_models = [k for k in accuracies.keys() if 'FS' in k]
    
    x = np.arange(len(base_models))
    width = 0.35
    
    plt.bar(x - width/2, [accuracies[m] for m in base_models], width, label='Base Model', color=colors[:len(base_models)])
    plt.bar(x + width/2, [accuracies[m] for m in fs_models], width, label='Feature Selection', color=colors[len(base_models):])
    
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison (2024 Season)')
    plt.xticks(x, [m.split()[0] for m in base_models], rotation=45)
    plt.legend()
    
    # Add value labels
    for i in x:
        if base_models[i] in accuracies:
            plt.text(i - width/2, accuracies[base_models[i]], f'{accuracies[base_models[i]]:.2%}', 
                    ha='center', va='bottom')
        if fs_models[i] in accuracies:
            plt.text(i + width/2, accuracies[fs_models[i]], f'{accuracies[fs_models[i]]:.2%}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('model_comparison', exist_ok=True)
    plt.savefig('model_comparison/accuracy_comparison.png')
    plt.close()
    
    # Print summary
    print("\nModel Accuracies:")
    print("================")
    for model, acc in accuracies.items():
        print(f"{model:15s}: {acc:.2%}")

def main():
    print("F1 2024 Season Model Comparison")
    print("===============================")
    
    # Create output directory
    if not os.path.exists('model_comparison'):
        os.makedirs('model_comparison')
    
    # Generate accuracy visualization
    print("\nGenerating model accuracy comparison...")
    visualize_model_accuracy()
    
    print("\nVisualization saved to model_comparison/accuracy_comparison.png")

if __name__ == "__main__":
    main()