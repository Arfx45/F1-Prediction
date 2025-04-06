import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from main import compare_2024_predictions

def visualize_model_comparison():
    # Create visualization directory if it doesn't exist
    if not os.path.exists('model_comparison'):
        os.makedirs('model_comparison')
    
    # Get comparison results
    accuracies, predictions = compare_2024_predictions()
    
    # Convert to DataFrame for easier plotting
    results_df = pd.DataFrame({
        'Model': list(accuracies.keys()),
        'Accuracy': list(accuracies.values())
    })
    
    # 1. Accuracy Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Model', y='Accuracy')
    plt.title('Model Accuracy Comparison for 2024 F1 Season')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison/accuracy_comparison.png')
    plt.close()
    
    # 2. Create detailed comparison table
    comparison_table = pd.DataFrame({
        'Model': list(accuracies.keys()),
        'Accuracy': [f"{acc:.2%}" for acc in accuracies.values()],
        'Predictions Made': [len(pred) for pred in predictions.values()]
    })
    
    # Save comparison table
    comparison_table.to_csv('model_comparison/model_comparison.csv', index=False)
    
    # Print results
    print("\nDetailed Model Comparison")
    print("=======================")
    print(comparison_table.to_string(index=False))
    
    return comparison_table

if __name__ == "__main__":
    visualize_model_comparison()