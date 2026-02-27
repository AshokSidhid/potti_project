import numpy as np

def train_energy_model():
    """
    Returns the physical constants required for the Tractive Force Equation.
    Designed for native execution in Python 3.14t.
    """
    return {
        "mass_kg": 1600.0,         # Average EV mass (e.g., standard hatchback)
        "frontal_area_m2": 2.2,    # Frontal area hitting the wind
        "drag_coeff": 0.28,        # Aerodynamic drag coefficient
        "rolling_res": 0.012,      # Tire friction coefficient on asphalt
        "regen_efficiency": 0.65   # Battery recovery efficiency on descents
    }

def predict_energy_dynamic(length_meters: float, speed_kph: float, slope_percent: float, coeffs: dict = None) -> float:
    """
    Calculates EV energy using real-world tractive physics.
    """
    if coeffs is None:
        coeffs = train_energy_model()
        
    # Standard environmental constants
    speed_mps = speed_kph / 3.6
    gravity = 9.81
    air_density = 1.225
    
    # Calculate slope angle in radians
    theta = np.arctan(slope_percent / 100.0)
    
    # 1. Rolling Resistance Force
    f_roll = coeffs["rolling_res"] * coeffs["mass_kg"] * gravity * np.cos(theta)
    
    # 2. Gradient Force (Uphill = positive drag, Downhill = gravity assist)
    f_grad = coeffs["mass_kg"] * gravity * np.sin(theta)
    
    # 3. Aerodynamic Drag Force (Quadratic impact from speed)
    f_aero = 0.5 * air_density * coeffs["drag_coeff"] * coeffs["frontal_area_m2"] * (speed_mps ** 2)
    
    # Total Tractive Force (Newtons)
    total_force = f_roll + f_grad + f_aero
    
    # Energy = Force x Distance (Joules)
    energy_joules = total_force * length_meters
    
    # Convert Joules to standard EV Watt-hours (Wh)
    energy_wh = energy_joules / 3600.0
    
    # Apply regenerative braking if the vehicle is coasting downhill
    if energy_wh < 0:
        energy_wh = energy_wh * coeffs["regen_efficiency"]
        
    # Prevent extreme anomalies from breaking the A* routing logic
    return max(energy_wh, -length_meters * 0.05)