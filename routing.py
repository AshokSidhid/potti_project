import networkx as nx
import math
from ml_model import predict_energy_dynamic

def heuristic_energy(node_u, node_v, G):
    """
    The A* Heuristic: Estimates the minimum possible energy to the destination.
    Uses straight-line distance and ideal flat-ground conditions.
    """
    lat1, lon1 = G.nodes[node_u]['y'], G.nodes[node_u]['x']
    lat2, lon2 = G.nodes[node_v]['y'], G.nodes[node_v]['x']
    
    # Haversine formula for straight-line distance in meters
    radius = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_m = radius * c
    
    # Assume absolute best case scenario: 40 km/h, perfectly flat (0 slope)
    return predict_energy_dynamic(distance_m, speed_kph=40.0, slope_percent=0.0)

def dynamic_energy_weight(u, v, edge_data, model_coeffs=None):
    """
    Lazy Evaluation: Calculates energy ON DEMAND only for the streets A* explores.
    """
    data = edge_data[0] if isinstance(edge_data, dict) and 0 in edge_data else edge_data
    
    length_m = data.get('length', 1.0)
    speed = data.get('speed_kph', 40.0)
    
    # --- THE CRITICAL MATH FIX ---
    # OSMnx provides grade as a raw decimal (e.g., 0.05). 
    # Our physics model tractive equation needs a percentage (e.g., 5.0).
    grade_decimal = data.get('grade', 0.0) 
    slope_percent = grade_decimal * 100.0
    
    return predict_energy_dynamic(length_m, speed, slope_percent, model_coeffs)

def find_energy_route_astar(G, origin_node, destination_node, model_coeffs=None):
    """
    Finds the minimum energy route using A* and dynamic physics/ML weights.
    """
    return nx.astar_path(
        G, 
        origin_node, 
        destination_node, 
        heuristic=lambda u, v: heuristic_energy(u, v, G),
        weight=lambda u, v, d: dynamic_energy_weight(u, v, d, model_coeffs)
    )

def find_shortest_route(G, origin_node, destination_node):
    """Finds the absolute shortest physical distance (Dijkstra)."""
    return nx.shortest_path(G, origin_node, destination_node, weight='length')