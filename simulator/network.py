import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import os

# Configure OSMnx to make downloading reliable
ox.settings.log_console = True
ox.settings.use_cache = True

def download_campus_network(place_name="Visvesvaraya National Institute of Technology, Nagpur, India"):
    """
    Downloads the street network of VNIT Nagpur from OpenStreetMap.
    We use a custom filter to ensure we get all drivable roads (including "service" roads
    inside the campus, which are normally filtered out by standard "drive" queries).
    If the Nominatim place polygon is too restrictive (causing 0 roads to be clipped),
    we fall back to a GPS bounding box centered on VNIT Nagpur.
    """
    print(f"Attempting to download street network for: {place_name}...")
    
    # Custom filter to get public streets AND university campus service roads, excluding footways/footpaths
    custom_highway_filter = '["highway"~"primary|secondary|tertiary|residential|service|unclassified"]'
    
    try:
        # Try downloading by place name first
        G = ox.graph_from_place(
            place_name, 
            custom_filter=custom_highway_filter
        )
        
        # If the graph is empty or has no edges, force fallback
        if len(G.edges) == 0:
            raise ValueError("Place polygon returned an empty network.")
            
    except Exception as e:
        print(f"Place query failed or returned empty graph: {e}")
        print("Falling back to GPS Bounding Box download for VNIT Nagpur campus...")
        
        # Bounding box coordinates covering the VNIT Nagpur campus rectangle:
        # Latitude range: 21.118 to 21.135 N
        # Longitude range: 79.040 to 79.062 E
        G = ox.graph_from_bbox(
            north=21.135,
            south=21.118,
            east=79.062,
            west=79.040,
            custom_filter=custom_highway_filter
        )
    
    # Store original WGS84 latitude/longitude before projection drops/renames them
    for node, data in G.nodes(data=True):
        data['orig_lat'] = data.get('lat', data.get('y', 0.0))
        data['orig_lon'] = data.get('lon', data.get('x', 0.0))
        
    # Clean the graph by converting it to a standard projected coordinate system (meters)
    G = ox.project_graph(G)
    
    print(f"Original Network Ingested: {len(G.nodes)} intersections and {len(G.edges)} road segments.")
    return G

def transform_to_dual_graph(G):
    """
    Applies the Line Graph (Dual Graph) transformation.
    - Each ROAD SEGMENT (edge in G) becomes a NODE in our new Dual Graph.
    - Connecting INTERSECTIONS (nodes in G) become EDGES in our new Dual Graph.
    """
    print("Transforming road network to Dual Graph (Line Graph)...")
    
    # nx.line_graph automatically transforms edges of G to nodes of L(G)
    # Since G is a MultiDiGraph, we convert it to a standard DiGraph (directed graph) first
    G_simple = nx.DiGraph(G)
    L = nx.line_graph(G_simple)
    
    # Store critical metadata on our new Dual Nodes (which represent physical roads)
    # We calculate the geometric center (centroid) of each road so we can map it later
    for u, v in L.nodes:
        # (u, v) is a road segment in the original graph starting at intersection 'u' and ending at 'v'
        orig_node_u = G.nodes[u]
        orig_node_v = G.nodes[v]
        
        # Midpoint calculation (average coordinates of the two intersections)
        # x = longitude/easting, y = latitude/northing (in UTM meters for NetworkX layout)
        centroid_x = (orig_node_u['x'] + orig_node_v['x']) / 2
        centroid_y = (orig_node_u['y'] + orig_node_v['y']) / 2
        
        # Also compute WGS84 coordinates (latitude/longitude in degrees) for Folium mapping
        centroid_lat = (orig_node_u['orig_lat'] + orig_node_v['orig_lat']) / 2
        centroid_lon = (orig_node_u['orig_lon'] + orig_node_v['orig_lon']) / 2
        
        # Save these properties to our Dual Graph Node
        L.nodes[(u, v)]['centroid_x'] = centroid_x
        L.nodes[(u, v)]['centroid_y'] = centroid_y
        L.nodes[(u, v)]['centroid_lat'] = centroid_lat
        L.nodes[(u, v)]['centroid_lon'] = centroid_lon
        
        # Save start and end coordinates of the road segment
        L.nodes[(u, v)]['start_lat'] = orig_node_u['orig_lat']
        L.nodes[(u, v)]['start_lon'] = orig_node_u['orig_lon']
        L.nodes[(u, v)]['end_lat'] = orig_node_v['orig_lat']
        L.nodes[(u, v)]['end_lon'] = orig_node_v['orig_lon']
        
        # Extract road length and name from original graph if available
        edge_data = G.get_edge_data(u, v)
        if edge_data and 0 in edge_data:
            L.nodes[(u, v)]['length'] = edge_data[0].get('length', 100.0)
            
            # Retrieve street name from OpenStreetMap attributes
            name_val = edge_data[0].get('name', 'Unnamed Campus Road')
            if isinstance(name_val, list):
                name_val = name_val[0]
            L.nodes[(u, v)]['name'] = str(name_val)
        else:
            L.nodes[(u, v)]['length'] = 100.0
            L.nodes[(u, v)]['name'] = 'Unnamed Campus Road'
            
        # Initialize default pavement features we will use later in the simulation
        L.nodes[(u, v)]['PCI'] = 100.0  # Perfect starting health
        L.nodes[(u, v)]['AADT'] = 1000.0  # Traffic volume
        L.nodes[(u, v)]['drainage_quality'] = 1.0  # Excellent drainage by default
        
    print(f"Dual Graph Completed: {len(L.nodes)} road nodes and {len(L.edges)} intersection links.")
    return L

def plot_networks(G, L, output_dir="data"):
    """
    Plots the physical street network and our transformed Dual Graph side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: The real-world campus map layout (Primal Graph)
    axes[0].set_title("1. VNIT Nagpur Physical Layout (Intersections & Roads)", fontsize=12)
    ox.plot_graph(G, ax=axes[0], node_color="blue", node_size=20, edge_color="gray", show=False, close=False)
    
    # Plot 2: The transformed Dual Graph (Roads as Nodes, Intersections as Links)
    axes[1].set_title("2. Transformed Dual Graph (Roads = Nodes, Links = Connections)", fontsize=12)
    
    # Position nodes using their calculated midpoints
    pos = {node: (data['centroid_x'], data['centroid_y']) for node, data in L.nodes(data=True)}
    nx.draw_networkx_nodes(L, pos, ax=axes[1], node_size=15, node_color="red")
    nx.draw_networkx_edges(L, pos, ax=axes[1], edge_color="black", alpha=0.3, arrows=True)
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "network_topology.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Visualizations saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Test script: Download, Transform, and Plot
    G = download_campus_network()
    L = transform_to_dual_graph(G)
    plot_networks(G, L)
