import networkx as nx
import osmnx as ox

def heuristic_energy(u, v, graph):
    """Ultra-fast A* heuristic: straight-line distance * min possible Wh/m."""
    u_data = graph.nodes[u]
    v_data = graph.nodes[v]
    distance_m = ox.distance.great_circle(u_data['y'], u_data['x'], v_data['y'], v_data['x'])
    
    # Assume a highly efficient 0.1 Wh per meter as the optimistic heuristic cost
    return distance_m * 0.1

def find_energy_route_astar(graph, orig_node, dest_node):
    """Finds the most energy-efficient route using pre-calculated ML edge weights."""
    return nx.astar_path(
        graph, 
        orig_node, 
        dest_node, 
        heuristic=lambda u, v: heuristic_energy(u, v, graph),
        weight='ml_energy_cost'  # Reads the pre-calculated batch prediction!
    )
    
def find_shortest_route(graph, orig_node, dest_node):
    """Finds the shortest physical route using standard Dijkstra."""
    return nx.shortest_path(graph, orig_node, dest_node, weight='length')