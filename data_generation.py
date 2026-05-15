import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=1000):
    """
    Generates a synthetic dataset for Rice Blast Disease Prediction.
    Features:
    - Temperature (Celsius): Optimal for disease 25-28 C.
    - Humidity (%): Optimal > 85%.
    - Leaf_Wetness (Hours): Optimal > 12 hours.
    - Lesion_Size (mm): Larger indicates disease.
    - Nitrogen_Level (mg/kg): High nitrogen can increase susceptibility.
    
    Target:
    - Status: 0 (Healthy), 1 (Rice Blast Disease) - Imbalanced dataset.
    """
    np.random.seed(42)
    
    # Generate features for Healthy plants (Class 0)
    # Roughly 850 samples
    num_healthy = int(num_samples * 0.85)
    temp_h = np.random.normal(22, 3, num_healthy)
    humidity_h = np.random.normal(70, 10, num_healthy)
    leaf_wet_h = np.random.normal(8, 3, num_healthy)
    lesion_h = np.random.exponential(1.5, num_healthy) # small lesions or spots
    nitrogen_h = np.random.normal(40, 10, num_healthy)
    status_h = np.zeros(num_healthy, dtype=int)
    
    # Generate features for Diseased plants (Class 1)
    # Roughly 150 samples
    num_diseased = num_samples - num_healthy
    temp_d = np.random.normal(26.5, 2, num_diseased) # Sweet spot for disease
    humidity_d = np.random.normal(90, 5, num_diseased) # High humidity
    leaf_wet_d = np.random.normal(14, 2, num_diseased) # Prolonged wetness
    lesion_d = np.random.normal(5, 2, num_diseased) # Larger lesions
    nitrogen_d = np.random.normal(60, 15, num_diseased) # Higher nitrogen
    status_d = np.ones(num_diseased, dtype=int)
    
    # Combine
    Temperature = np.concatenate([temp_h, temp_d])
    Humidity = np.concatenate([humidity_h, humidity_d])
    Leaf_Wetness = np.concatenate([leaf_wet_h, leaf_wet_d])
    Lesion_Size = np.concatenate([lesion_h, lesion_d])
    Nitrogen_Level = np.concatenate([nitrogen_h, nitrogen_d])
    Status = np.concatenate([status_h, status_d])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': Temperature,
        'Humidity': Humidity,
        'Leaf_Wetness': Leaf_Wetness,
        'Lesion_Size': Lesion_Size,
        'Nitrogen_Level': Nitrogen_Level,
        'Status': Status
    })
    
    # Ensure no negative values where they don't make sense
    df['Leaf_Wetness'] = df['Leaf_Wetness'].clip(lower=0)
    df['Lesion_Size'] = df['Lesion_Size'].clip(lower=0)
    df['Nitrogen_Level'] = df['Nitrogen_Level'].clip(lower=0)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic Rice Blast dataset...")
    df = generate_synthetic_data(1000)
    
    # Save the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'rice_blast_data.csv')
    df.to_csv(file_path, index=False)
    
    print(f"Dataset saved to {file_path}")
    print("\nClass Distribution:")
    print(df['Status'].value_counts(normalize=True) * 100)
    print("\nData Preview:")
    print(df.head())
