import numpy as np

def train_energy_model():
    """
    In a real-world scenario, this function would load historical EV 
    telematics data to train an XGBoost or Random Forest model. 
    For this prototype, we will return physics coefficients for an average EV (like a Nissan Leaf).
    """
    # [Mass (kg), Frontal Area (m2), Drag Coefficient, Rolling Resistance]
    ev_physics_coeffs = {
        "mass_kg": 1500,  
        "frontal_area_m2": 2.2,
        "drag_coeff": 0.28,
        "rolling_res": 0.012
    }
    return ev_physics_coeffs

def predict_energy_dynamic(length_meters: float, speed_kph: float, slope_percent: float, coeffs: dict = None) -> float:
    """
    Calculates EV energy using real-world tractive physics, replacing the dummy data.
    """
    if coeffs is None:
        coeffs = train_energy_model()
        
    speed_mps = speed_kph / 3.6
    gravity = 9.81
    air_density = 1.225
    
    # Calculate slope angle in radians
    theta = np.arctan(slope_percent / 100.0)
    
    # 1. Rolling Resistance Force
    f_roll = coeffs["rolling_res"] * coeffs["mass_kg"] * gravity * np.cos(theta)
    
    # 2. Gradient Force (Uphill takes power, downhill gives power)
    f_grad = coeffs["mass_kg"] * gravity * np.sin(theta)
    
    # 3. Aerodynamic Drag Force
    f_aero = 0.5 * air_density * coeffs["drag_coeff"] * coeffs["frontal_area_m2"] * (speed_mps ** 2)
    
    # Total Tractive Force (Newtons)
    total_force = f_roll + f_grad + f_aero
    
    # Energy = Force x Distance (Joules)
    energy_joules = total_force * length_meters
    
    # Convert Joules to Watt-hours (Wh) for standard EV metrics
    energy_wh = energy_joules / 3600.0
    
    # Account for regenerative braking (capped at 60% efficiency on downhills)
    if energy_wh < 0:
        energy_wh = energy_wh * 0.60
        
    # Prevent extreme negative values on cliffs
    return max(energy_wh, -length_meters * 0.05)