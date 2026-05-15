import os
import joblib
import pandas as pd

def predict_new_data(sample_data):
    """
    Loads the trained MLP model and Scaler, and makes predictions on new data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'mlp_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found! Please run train_model.py first.")
        return
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Convert input list to DataFrame matching the training features
    feature_names = ['Temperature', 'Humidity', 'Leaf_Wetness', 'Lesion_Size', 'Nitrogen_Level']
    df_sample = pd.DataFrame(sample_data, columns=feature_names)
    
    # Preprocess
    scaled_data = scaler.transform(df_sample)
    
    # Predict
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    
    # Display Results
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "Rice Blast Disease Detected" if pred == 1 else "Healthy"
        confidence = prob[pred] * 100
        print(f"Sample {i+1}:")
        print(f"  Inputs: {sample_data[i]}")
        print(f"  Prediction: {status}")
        print(f"  Confidence: {confidence:.2f}%\n")

if __name__ == "__main__":
    # Sample 1: Typical Healthy conditions
    # Sample 2: Typical Diseased conditions (High Temp, High Humidity, High Leaf Wetness, Large Lesions)
    new_samples = [
        [22.5, 71.0, 7.5, 1.2, 45.0],  # Likely Healthy
        [27.0, 92.0, 14.5, 5.5, 65.0]  # Likely Diseased
    ]
    
    print("Making predictions on new samples...\n")
    predict_new_data(new_samples)
