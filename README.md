# AI-Based Electric Vehicle Route Optimization System

![Python Version](https://img.shields.io/badge/python-3.14t-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-A%2A%20Search-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![API](https://img.shields.io/badge/Maps-Google%20Tiles-4285F4)

## 📌 Project Overview
Traditional navigation systems prioritize the shortest distance or fastest time, often neglecting the specific energy constraints of Electric Vehicles (EVs). This project implements an intelligent routing engine that optimizes for **minimum energy consumption** rather than just distance.

By integrating real-world topographical data (elevation), aerodynamic drag coefficients, and rolling resistance, the system reduces "range anxiety" and provides dynamic re-routing to charging stations when battery levels are insufficient.



## 🚀 Key Features
* **A* Search Algorithm:** Uses informed heuristics to find energy-optimal paths, proactively avoiding steep inclines.
* **Physics-Informed Energy Model:** Calculates the **Tractive Force Equation** ($F_{total} = F_{roll} + F_{grad} + F_{aero}$) to determine real-world Watt-hour (Wh) costs.
* **Live Topographical Integration:** Fetches actual elevation data from the **OpenTopoData API** to calculate road grades.
* **Dynamic Charging Stop Logic:** Automatically detects low battery and re-routes the vehicle to the nearest real-world charging station using **OpenStreetMap (OSMnx)** data.
* **Google Maps Visualization:** Features a high-detail interactive dashboard using Google Maps tile layers for professional presentation.

## 🛠️ Technical Stack
* **Language:** Python 3.14t (Compatible with free-threaded execution)
* **Graph Processing:** NetworkX, OSMnx
* **Web Framework:** Streamlit
* **Visualization:** Folium, Streamlit-Folium
* **Data Sources:** OpenStreetMap, OpenTopoData API, Google Maps Tiles

## 📥 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AshokSidhid/potti_project.git](https://github.com/AshokSidhid/potti_project.git)
   cd potti_project
   ```


2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Methodology

### 1. The Energy Model

The system "trains" on physical vehicle constants (Mass, Drag Coefficient, Frontal Area) to predict energy consumption on any given road segment:

$$Energy (Wh) = \frac{(F_{rolling} + F_{gradient} + F_{aerodynamic}) \times Distance}{3600}$$

### 2. A* vs Dijkstra Comparison

* **Dijkstra (Shortest Path):** Minimizes raw distance ($km$). In hilly terrain like Hyderabad, this often leads to massive battery drain by driving over steep inclines.
* **A* (Energy Efficient):** Minimizes energy cost ($Wh$). It utilizes a heuristic to identify detours that are physically longer but consume less battery by maintaining a flatter elevation profile.

### 3. Smart Charging Detours

If $Available Battery < Required Energy$, the system:

1. Locates all charging stations within a 10km radius of the route midpoint.
2. Selects the most efficient station.
3. Generates a new two-leg A* path: `Source -> Charging Station -> Destination`.

## 🤝 Contributors

* **Batch-04**
* **Lekhana G** (2300031636)
* **Madhuri** (2300032809)

---

*Note: This project is a functional prototype. In a production environment, real-time traffic telematics and high-resolution Digital Elevation Models (DEM) would be integrated.*

```

### Next Step
Since you mentioned being sleepy earlier, I've ensured this is the "final version" that matches all your latest edits. Would you like me to generate a **`requirements.txt`** file now so your friend has the complete package ready for her submission?
