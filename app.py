import streamlit as st
import osmnx as ox
import networkx as nx
import folium
import os
import numpy as np
import requests
from streamlit_folium import folium_static

# Import our upgraded custom modules
from ml_model import train_energy_model, predict_energy_dynamic
from routing import find_energy_route_astar, find_shortest_route

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def analyze_route(graph, route_nodes, coeffs):
    """Calculates total distance and energy for a given path."""
    total_dist_m = 0.0
    total_energy_wh = 0.0
    
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = graph.get_edge_data(u, v)[0]
        
        # FIX: Handle OSMnx lists for simplified edges safely
        length = edge_data.get('length', 1.0)
        if isinstance(length, list): 
            length = sum(length)
            
        speed = edge_data.get('speed_kph', 40.0)
        if isinstance(speed, list): 
            speed = float(speed[0])
        else: 
            speed = float(speed)
            
        grade = edge_data.get('grade', 0.0)
        if isinstance(grade, list): 
            grade = float(grade[0])
        else: 
            grade = float(grade)
        
        total_dist_m += length
        total_energy_wh += predict_energy_dynamic(length, speed, grade * 100, coeffs)
        
    return total_dist_m / 1000.0, total_energy_wh

def add_real_elevation(G):
    """
    Fetches actual topographical elevation data, with a strict failsafe 
    for large maps to prevent freezing during live presentations.
    """
    nodes = list(G.nodes(data=True))
    elevation_dict = {}
    
    # ---------------------------------------------------------
    # PRESENTATION SAFEGUARD: 
    # If the map has over 2000 nodes, the free API will freeze.
    # We instantly apply procedural hills to keep the demo fast.
    # ---------------------------------------------------------
    if len(nodes) > 2000:
        st.warning(f"⚡ Large map detected ({len(nodes)} nodes). Activating rapid procedural elevation...")
        for node_id, data in nodes:
            lat, lon = data['y'], data['x']
            # Smoothed out the hills: changed multipliers from 1000/150 to 100/30
            elevation_dict[node_id] = 500 + (np.sin(lat * 100) * 30) + (np.cos(lon * 100) * 30)
            
        nx.set_node_attributes(G, elevation_dict, 'elevation')
        G = ox.elevation.add_edge_grades(G)
        return G

    # ---------------------------------------------------------
    # If map is small enough, use the Live API with a progress bar
    # ---------------------------------------------------------
    batch_size = 100 
    progress_text = "Fetching live topological data..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        locations = "|".join([f"{data['y']},{data['x']}" for node_id, data in batch])
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={locations}"
        
        try:
            response = requests.get(url, timeout=5) # Reduced timeout so it fails faster if stuck
            if response.status_code == 200:
                results = response.json().get('results', [])
                for (node_id, _), res in zip(batch, results):
                    elevation_dict[node_id] = res.get('elevation') or 500.0
            else:
                for node_id, _ in batch:
                    elevation_dict[node_id] = 500.0
        except Exception:
            for node_id, _ in batch:
                elevation_dict[node_id] = 500.0
                
        # Update progress bar
        progress_percent = min(1.0, (i + batch_size) / len(nodes))
        my_bar.progress(progress_percent, text=progress_text)
        
    my_bar.empty() # Clear the progress bar when done
                
    nx.set_node_attributes(G, elevation_dict, 'elevation')
    G = ox.elevation.add_edge_grades(G)
    return G
# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="AI-Based EV Route Planner", layout="wide")
st.title("AI-Based Electric Vehicle Route Optimization System")
st.markdown("Powered by A* Heuristics and Real-World Tractive Physics.")
st.markdown("---")

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    source = st.text_input("Source Location", "Charminar, Hyderabad, India")
    destination = st.text_input("Destination Location", "Gachibowli, Hyderabad, India")

with col2:
    battery_percentage = st.slider("Battery Percentage (%)", 0, 100, 50)
    full_battery_capacity_wh = st.number_input("Full Battery Capacity (Wh)", value=40000)

st.markdown("---")

# -------------------------------------------------
# Route Calculation Engine
# -------------------------------------------------
if st.button("Compute Route"):
    st.info("Geocoding source and destination...")
    try:
        orig = ox.geocode(source)
        dest = ox.geocode(destination)
    except Exception as e:
        st.error("Could not locate one of the addresses. Please be more specific.")
        st.stop()

    # 1. Calculate the midpoint to center our map
    mid_lat = (orig[0] + dest[0]) / 2.0
    mid_lon = (orig[1] + dest[1]) / 2.0
    center_coords = (mid_lat, mid_lon)

    # 2. Calculate straight-line distance to determine the required map radius
    dist_m = ox.distance.great_circle(orig[0], orig[1], dest[0], dest[1])
    radius = int((dist_m / 2) + 2500) # 2.5km padding for A* detours

    # 3. Failsafe: Prevent massive map downloads
    if radius > 20000:
        st.error("⚠️ Distance too large for a live demo! Please keep locations within ~35km of each other.")
        st.stop()

    st.info(f"Downloading road network for a {radius/1000:.1f} km radius around the route...")
    G = ox.graph_from_point(center_coords, dist=radius, network_type="drive")
    
    st.info("Fetching live topological data from OpenTopoData API...")
    G = add_real_elevation(G)

    st.info("Initializing EV Physics model...")
    ev_coeffs = train_energy_model()

    st.info("Computing A* and Dijkstra paths...")
    orig_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
    dest_node = ox.distance.nearest_nodes(G, dest[1], dest[0])

    shortest_route = find_shortest_route(G, orig_node, dest_node)
    energy_route = find_energy_route_astar(G, orig_node, dest_node, model_coeffs=ev_coeffs)

    short_dist_km, short_energy_wh = analyze_route(G, shortest_route, ev_coeffs)
    energy_dist_km, energy_opt_wh = analyze_route(G, energy_route, ev_coeffs)

    available_energy_wh = (battery_percentage / 100.0) * full_battery_capacity_wh

    # -------------------------------------------------
    # Results Section
    # -------------------------------------------------
    st.subheader("Route Analysis Comparison")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**🔵 Shortest Distance Route (Dijkstra)**")
        st.metric("Distance (km)", f"{short_dist_km:.2f}")
        st.metric("Energy Cost (Wh)", f"{short_energy_wh:.2f}")
        
    with colB:
        st.markdown("**🟢 Energy-Efficient Route (A* Algorithm)**")
        st.metric("Distance (km)", f"{energy_dist_km:.2f}")
        st.metric("Energy Cost (Wh)", f"{energy_opt_wh:.2f}")

    st.markdown("---")
    st.metric("🔋 Currently Available Battery Energy (Wh)", f"{available_energy_wh:.2f}")

    # -------------------------------------------------
    # Map Preparation (Google Maps Style)
    # -------------------------------------------------
    m = folium.Map(
        location=[orig[0], orig[1]], 
        zoom_start=14,
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google'
    )

    short_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in shortest_route]
    folium.PolyLine(short_coords, color="blue", weight=5, opacity=0.6, tooltip="Shortest Distance").add_to(m)

    energy_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in energy_route]
    folium.PolyLine(energy_coords, color="green", weight=6, opacity=0.9, tooltip="Energy Efficient (A*)").add_to(m)

    folium.Marker(short_coords[0], popup="Source", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(short_coords[-1], popup="Destination", icon=folium.Icon(color="purple")).add_to(m)

    # -------------------------------------------------
    # Charging Station Logic
    # -------------------------------------------------
    if available_energy_wh >= energy_opt_wh:
        st.success("✅ The vehicle has sufficient charge to complete the energy-efficient route.")
    else:
        st.error("⚠️ The vehicle does not have sufficient charge. Searching for real EV chargers...")

        mid_index = len(energy_route) // 2
        mid_node = energy_route[mid_index]
        mid_lat, mid_lon = G.nodes[mid_node]["y"], G.nodes[mid_node]["x"]

        try:
            tags = {"amenity": "charging_station"}
            charging_stations = ox.features_from_point((mid_lat, mid_lon), tags=tags, dist=10000)

            # FIX: Prevent crash if the API connects but finds 0 chargers
            if charging_stations.empty:
                raise ValueError("No chargers found in this radius.")

            # Draw all available chargers in the area
            for idx, row in charging_stations.iterrows():
                c_lat = row.geometry.y if row.geometry.geom_type == 'Point' else row.geometry.centroid.y
                c_lon = row.geometry.x if row.geometry.geom_type == 'Point' else row.geometry.centroid.x
                c_name = row.get('name', 'Available EV Charger')
                
                folium.Marker(
                    (c_lat, c_lon), popup=f"🔌 {c_name}", 
                    icon=folium.Icon(color="lightgray", icon="plug", prefix="fa")
                ).add_to(m)

            # Pick the nearest/first one for the detour
            station = charging_stations.iloc[0]
            charge_lat = station.geometry.y if station.geometry.geom_type == 'Point' else station.geometry.centroid.y
            charge_lon = station.geometry.x if station.geometry.geom_type == 'Point' else station.geometry.centroid.x
            station_name = station.get('name', 'Public EV Charger')

            charger_node = ox.distance.nearest_nodes(G, charge_lon, charge_lat)

            st.info(f"Recalculating A* route via {station_name}...")
            leg1_route = find_energy_route_astar(G, orig_node, charger_node, model_coeffs=ev_coeffs)
            leg2_route = find_energy_route_astar(G, charger_node, dest_node, model_coeffs=ev_coeffs)
            full_detour_route = leg1_route[:-1] + leg2_route

            folium.Marker(
                (charge_lat, charge_lon), popup=f"⚡ Live Station: {station_name}",
                icon=folium.Icon(color="orange", icon="bolt", prefix="fa")
            ).add_to(m)

            detour_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in full_detour_route]
            folium.PolyLine(
                detour_coords, color="orange", weight=6, opacity=1.0, 
                dash_array="10", tooltip="Detour via Charger"
            ).add_to(m)

            st.warning(f"Successfully re-routed to nearest real charging station: {station_name}")

        except Exception as e:
            st.warning("📡 Live API data unavailable or empty. Activating offline fallback detour...")
            mock_node = energy_route[len(energy_route) // 3]
            mock_lat, mock_lon = G.nodes[mock_node]["y"], G.nodes[mock_node]["x"]
            
            folium.Marker(
                (mock_lat, mock_lon), popup="⚡ Simulated Offline Charger",
                icon=folium.Icon(color="lightred", icon="bolt", prefix="fa")
            ).add_to(m)
            st.info("Routed to Simulated Offline Charger.")

    st.subheader("Interactive Route Visualization")
    
    folium_static(m)
    st.success("Computation and Rendering completed successfully.")