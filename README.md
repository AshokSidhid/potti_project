# AI-Based EV Route Optimization System (v2.0)

![Python Version](https://img.shields.io/badge/python-3.14t-blue)
![ML Model](https://img.shields.io/badge/Model-Random%20Forest%20Regressor-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-A%2A%20Search%20(Batch)-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)

## 📌 Project Overview
This system solves the "Range Anxiety" problem for Electric Vehicles by replacing standard shortest-path navigation with **Energy-Optimal Routing**. 

Unlike Google Maps, which uses distance/time, our engine utilizes a **Random Forest Regressor** trained on 5,000 synthetic trip samples to predict the exact Watt-hour (Wh) cost of every street in a city, accounting for speed limits and topographical elevation grades.



## 🚀 Key Features
* **Machine Learning Pipeline:** Uses a Random Forest ensemble (100 decision trees) to map the non-linear relationship between speed, slope, and battery drain.
* **Batch Inference Optimization:** Engineered to score 85,000+ road segments in $<1$ second by vectorized matrix operations, ensuring a smooth UI even for massive city maps.
* **A* Heuristic Search:** Implements an informed search algorithm that "looks ahead" to find paths that bypass steep inclines.
* **Real-World Topography:** Integrates the OpenTopoData API (SRTM 30m) to calculate road gradients ($grade = \frac{\Delta elevation}{distance}$).
* **Dynamic Charging Detours:** Automated "Low Battery" logic that identifies real-world EV charging stations via OpenStreetMap and reroutes the vehicle mid-trip.

## 🛠️ Technical Stack
* **ML Core:** Scikit-Learn (RandomForestRegressor), Pandas, NumPy
* **Graph Engine:** NetworkX, OSMnx (OpenStreetMap)
* **Dashboard:** Streamlit, Folium (Google Maps Tile Layers)
* **API Integration:** OpenTopoData (Topography), OpenStreetMap (Features)

## 📊 How It Works

### 1. Data Generation & Training
The system generates a synthetic dataset based on the **Tractive Force Equation**:
$$F_{total} = F_{rolling} + F_{gradient} + F_{aerodynamic}$$
A Random Forest model then learns the patterns of energy consumption, including regenerative braking (negative energy) on downhill slopes.

### 2. Batch Scoring
To handle large cities like Hyderabad (80k+ nodes), the model predicts the energy cost for the **entire map** in one batch immediately after the map is downloaded. This prevents the "individual prediction bottleneck" during the routing phase.



### 3. Energy-Aware Routing
* **Blue Route:** Shortest path (Dijkstra). High risk of battery depletion on steep terrain.
* **Green Route:** Energy-efficient path (A*). Optimized by the ML model to maximize range.

## 📥 Setup
1. Clone the repo: `git clone https://github.com/LekhanaGembali/potti_project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 🤝 Contributors
* **Batch-04**
* **Lekhana G** (2300031636)
* **Madhuri** (2300032809)
