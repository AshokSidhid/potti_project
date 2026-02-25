import osmnx as ox
import networkx as nx
import folium
import numpy as np

# Import our newly upgraded modules
from ml_model import train_energy_model, predict_energy_dynamic
from routing import find_energy_route_astar, find_shortest_route

print("Downloading Hyderabad road network...")
# We use a smaller subset (e.g., Banjara Hills) for faster prototype testing.
# For the full city, change back to "Hyderabad, Telangana, India"
place_name = "Banjara Hills, Hyderabad, Telangana, India"
G = ox.graph_from_place(place_name, network_type="drive")
print("Graph downloaded successfully!")

# -------------------------------------------------
# STEP 1: Add Elevation and Slope Data
# -------------------------------------------------
print("Simulating topological elevation data...")

# Procedural elevation: Generates realistic rolling hills between 500m and 600m altitude
# In a production environment, you would use a local DEM (Digital Elevation Model) .tif file here.
elevation_dict = {}
for node, data in G.nodes(data=True):
    lat, lon = data['y'], data['x']
    # Create a mathematical "wave" based on coordinates to simulate hills
    simulated_elevation = 500 + (np.sin(lat * 1000) * 50) + (np.cos(lon * 1000) * 50)
    elevation_dict[node] = simulated_elevation

# Map elevations to nodes
nx.set_node_attributes(G, elevation_dict, 'elevation')

# Automatically calculate the 'grade' (slope) for every street
# osmnx uses the rise/run formula between the connected nodes' elevations
G = ox.elevation.add_edge_grades(G)

# -------------------------------------------------
# STEP 2: Locations
# -------------------------------------------------
print("Geocoding locations...")
origin = ox.geocode("KBR Park, Hyderabad, India")
destination = ox.geocode("City Center Mall, Banjara Hills, Hyderabad, India")

orig_node = ox.distance.nearest_nodes(G, origin[1], origin[0])
dest_node = ox.distance.nearest_nodes(G, destination[1], destination[0])

# -------------------------------------------------
# STEP 3: Calculate Routes (Using Lazy Evaluation A*)
# -------------------------------------------------
print("Calculating shortest route (Dijkstra)...")
shortest_route = find_shortest_route(G, orig_node, dest_node)

print("Calculating energy-efficient route (A* + Physics)...")
# Load EV physics coefficients (Mass, Drag, etc.)
ev_coeffs = train_energy_model() 

# This now runs the fast A* search without pre-calculating the whole city!
energy_route = find_energy_route_astar(G, orig_node, dest_node, model_coeffs=ev_coeffs)

# -------------------------------------------------
# STEP 4: Compare Distance and Energy
# -------------------------------------------------
def analyze_route(graph, route_nodes, coeffs):
    """Helper to sum up distance and energy for a specific path."""
    total_dist_m = 0.0
    total_energy_wh = 0.0
    
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = graph.get_edge_data(u, v)[0]
        length = edge_data.get('length', 1.0)
        speed = edge_data.get('speed_kph', 40.0)
        grade = edge_data.get('grade', 0.0)
        
        total_dist_m += length
        total_energy_wh += predict_energy_dynamic(length, speed, grade * 100, coeffs)
        
    return total_dist_m / 1000.0, total_energy_wh

short_dist_km, short_energy_wh = analyze_route(G, shortest_route, ev_coeffs)
energy_dist_km, energy_opt_wh = analyze_route(G, energy_route, ev_coeffs)

print("\n----- ROUTE COMPARISON -----")
print(f"Shortest Route: {short_dist_km:.2f} km | Requires {short_energy_wh:.2f} Wh")
print(f"Energy Route:   {energy_dist_km:.2f} km | Requires {energy_opt_wh:.2f} Wh")

# -------------------------------------------------
# STEP 5: Create Map
# -------------------------------------------------
print("\nCreating map...")
m = folium.Map(location=origin, zoom_start=14)

# Plot Shortest Route (Blue)
shortest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in shortest_route]
folium.PolyLine(shortest_coords, color="blue", weight=5, opacity=0.8, tooltip="Shortest Route").add_to(m)

# Plot Energy Route (Green)
energy_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in energy_route]
folium.PolyLine(energy_coords, color="green", weight=5, opacity=0.8, tooltip="Energy Efficient Route").add_to(m)

folium.Marker(origin, popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(destination, popup="Destination", icon=folium.Icon(color="purple")).add_to(m)

m.save("ev_route_map.html")
print("Map saved as ev_route_map.html. Open it in a browser to view.")