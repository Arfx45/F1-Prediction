import pandas as pd

def extract_2024_data():
    # Read the races data
    races_df = pd.read_csv('data/races.csv')
    
    # Filter for 2024 season
    races_2024 = races_df[races_df['year'] == 2024]
    
    # Save to new CSV file
    races_2024.to_csv('data/2024/races2024.csv', index=False)
    
    # Load other relevant datasets and filter them
    results_df = pd.read_csv('data/results.csv')
    constructors_df = pd.read_csv('data/constructors.csv')
    drivers_df = pd.read_csv('data/drivers.csv')
    
    # Get race IDs for 2024
    race_ids_2024 = races_2024['raceId'].tolist()
    
    # Filter results for 2024 races (this will be empty for now as 2024 hasn't happened)
    results_2024 = results_df[results_df['raceId'].isin(race_ids_2024)]
    
    # Save filtered datasets
    results_2024.to_csv('data/2024/results2024.csv', index=False)
    constructors_df.to_csv('data/2024/constructors2024.csv', index=False)
    drivers_df.to_csv('data/2024/drivers2024.csv', index=False)
    
    print(f"Number of 2024 races extracted: {len(races_2024)}")
    print(f"Datasets saved in data/2024/ directory")
    
    return races_2024

if __name__ == "__main__":
    # Create 2024 directory if it doesn't exist
    import os
    if not os.path.exists('data/2024'):
        os.makedirs('data/2024')
    
    # Extract and save data
    races_2024 = extract_2024_data()
    
    # Display first few races of 2024 season
    print("\n2024 Race Calendar:")
    print(races_2024[['raceId', 'name', 'date', 'time']].to_string())