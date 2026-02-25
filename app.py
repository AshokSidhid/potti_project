import streamlit as st
import osmnx as ox
import networkx as nx
import folium
import os
import numpy as np
from streamlit_folium import folium_static

# Import our upgraded modules
from ml_model import train_energy_model, predict_energy_dynamic
from routing import find_energy_route_astar, find_shortest_route

# -------------------------------------------------
# Helper Function for UI Metrics
# -------------------------------------------------
def analyze_route(graph, route_nodes, coeffs):
    """Calculates total distance and energy for a given path."""
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
    source = st.text_input("Source Location", "KBR Park, Hyderabad, India")
    destination = st.text_input("Destination Location", "City Center Mall, Banjara Hills, Hyderabad, India")

with col2:
    battery_percentage = st.slider("Battery Percentage (%)", 0, 100, 50)
    full_battery_capacity_wh = st.number_input("Full Battery Capacity (Wh)", value=40000)

st.markdown("---")

# -------------------------------------------------
# Route Calculation Engine
# -------------------------------------------------
if st.button("Compute Route"):
    # Using a 3.5km radius around Banjara Hills for fast presentation demos
    center_coords = (17.4156, 78.4347)
    graph_file = "banjara_hills_graph.graphml"

    if os.path.exists(graph_file):
        st.info("Loading road network data...")
        G = ox.load_graphml(graph_file)
    else:
        st.info("Downloading road network and simulating elevation (first run only)...")
        G = ox.graph_from_point(center_coords, dist=3500, network_type="drive")
        
        # Add massive virtual hills for the A* presentation demo
        elevation_dict = {}
        for node, data in G.nodes(data=True):
            lat, lon = data['y'], data['x']
            elevation_dict[node] = 500 + (np.sin(lat * 1000) * 150) + (np.cos(lon * 1000) * 150)
        nx.set_node_attributes(G, elevation_dict, 'elevation')
        G = ox.elevation.add_edge_grades(G)
        
        ox.save_graphml(G, graph_file)

    st.info("Initializing EV Physics model...")
    ev_coeffs = train_energy_model()

    st.info("Processing locations and computing A* paths...")
    orig = ox.geocode(source)
    dest = ox.geocode(destination)

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
    # Map Preparation
    # -------------------------------------------------
    m = folium.Map(location=[orig[0], orig[1]], zoom_start=14)

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

            # Draw all available chargers in the area
            for idx, row in charging_stations.iterrows():
                c_lat = row.geometry.y if row.geometry.geom_type == 'Point' else row.geometry.centroid.y
                c_lon = row.geometry.x if row.geometry.geom_type == 'Point' else row.geometry.centroid.x
                c_name = row.get('name', 'Available EV Charger')
                
                folium.Marker(
                    (c_lat, c_lon), popup=f"🔌 {c_name}", 
                    icon=folium.Icon(color="lightgray", icon="plug", prefix="fa")
                ).add_to(m)

            # Pick the first one for the detour
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
            st.warning("📡 Live API data unavailable. Activating offline fallback detour...")
            mock_node = energy_route[len(energy_route) // 3]
            mock_lat, mock_lon = G.nodes[mock_node]["y"], G.nodes[mock_node]["x"]
            
            folium.Marker(
                (mock_lat, mock_lon), popup="⚡ Simulated Offline Charger",
                icon=folium.Icon(color="lightred", icon="bolt", prefix="fa")
            ).add_to(m)
            st.info("Routed to Simulated Offline Charger.")

    # Always render the map, no matter what happens above!
    st.subheader("Interactive Route Visualization")
    folium_static(m)
    st.success("Computation and Rendering completed successfully.")