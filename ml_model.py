import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings

# Suppress standard sklearn feature warnings for cleaner Streamlit output
warnings.filterwarnings('ignore')

def generate_synthetic_ev_data(num_samples=5000):
    """
    Generates realistic EV telemetry data using physics + random noise.
    This simulates having a dataset of thousands of past vehicle trips.
    """
    np.random.seed(42) # For reproducible results during presentations
    
    # 1. Generate random road conditions
    lengths = np.random.uniform(10.0, 2000.0, num_samples)    # Segment length in meters
    speeds = np.random.uniform(10.0, 80.0, num_samples)       # Speed in km/h
    slopes = np.random.uniform(-15.0, 15.0, num_samples)      # Gradient percentage
    
    # 2. Physics Baseline (Our previous Tractive Force logic)
    mass_kg = 1600.0
    gravity = 9.81
    air_density = 1.225
    drag_coeff = 0.28
    frontal_area = 2.2
    rolling_res = 0.012
    regen_eff = 0.65

    energy_wh_list = []
    
    for length, speed_kph, slope in zip(lengths, speeds, slopes):
        speed_mps = speed_kph / 3.6
        theta = np.arctan(slope / 100.0)
        
        f_roll = rolling_res * mass_kg * gravity * np.cos(theta)
        f_grad = mass_kg * gravity * np.sin(theta)
        f_aero = 0.5 * air_density * drag_coeff * frontal_area * (speed_mps ** 2)
        
        total_force = f_roll + f_grad + f_aero
        energy_wh = (total_force * length) / 3600.0
        
        if energy_wh < 0:
            energy_wh *= regen_eff
            
        # 3. Inject Real-World Noise (Traffic, AC usage, Wind gusts)
        # Adds up to 15% random variance to make the ML model work for it
        noise_factor = np.random.uniform(0.85, 1.15)
        final_energy = max(energy_wh * noise_factor, -length * 0.05)
        energy_wh_list.append(final_energy)
        
    # Compile into a DataFrame
    df = pd.DataFrame({
        'length_m': lengths,
        'speed_kph': speeds,
        'slope_percent': slopes,
        'energy_consumed_wh': energy_wh_list
    })
    
    return df

def train_energy_model():
    """
    The core Machine Learning training loop.
    Returns a fully trained Random Forest model.
    """
    # 1. Get the dataset
    df = generate_synthetic_ev_data(num_samples=5000) 
    
    # 2. Split into Features (X) and Target (y)
    X = df[['length_m', 'speed_kph', 'slope_percent']]
    y = df['energy_consumed_wh']
    
    # 3. Train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    return rf_model

def predict_energy_dynamic(length_meters: float, speed_kph: float, slope_percent: float, model) -> float:
    """
    Inference Engine: Uses the trained Random Forest to predict energy cost 
    for a specific road segment during the A* routing.
    """
    # If no model is passed, train one on the fly (failsafe)
    if model is None:
        model = train_energy_model()
        
    # Format the input exactly how scikit-learn expects it (a 2D array)
    features = np.array([[length_meters, speed_kph, slope_percent]])
    
    # The model predicts the Wh cost
    predicted_wh = model.predict(features)[0]
    
    return float(predicted_wh)