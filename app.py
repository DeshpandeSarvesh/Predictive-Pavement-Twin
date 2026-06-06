import os
import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx
import folium
import streamlit as st
from streamlit_folium import folium_static, st_folium
from gnn.model import PavementPredictorGNN
import simulator.physics as phys

# =====================================================================
# PHASE 5: DIGITAL TWIN STREAMLIT DASHBOARD
# =====================================================================
# This app loads our trained GNN and physics simulator, displaying an
# interactive map of VNIT Nagpur where users can stress-test roads.
# =====================================================================

st.set_page_config(layout="wide", page_title="Pavement Management Digital Twin")

st.title("🚧 Predictive Pavement Digital Twin & Spatial GNN Dashboard")
st.markdown("""
*Integrating Civil Infrastructure Mechanics (IRC:37) with Spatial Graph Neural Networks*  
**VNIT Nagpur Campus, India**
""")

# =====================================================================
# 1. LOAD DATASET AND MODELS
# =====================================================================
# LANDMARKS FOR SIMPLIFIED ROAD NAMES
# =====================================================================
LANDMARKS = {
    "Main Gate (East)": (21.1257, 79.0573),
    "North Gate": (21.1293, 79.0545),
    "Administrative Block": (21.1265, 79.0545),
    "Central Library": (21.1272, 79.0528),
    "Dept of Computer Science": (21.1278, 79.0495),
    "Mega Hostel Complex": (21.1215, 79.0490),
    "Girls Hostel Block": (21.1230, 79.0480),
    "Sports Ground": (21.1255, 79.0450),
    "VNIT Guest House": (21.1290, 79.0535),
    "Faculty Residential Area": (21.1280, 79.0440),
    "Mechanical & Civil Depts": (21.1282, 79.0520),
    "Campus Cafeteria": (21.1262, 79.0515)
}

def get_simplified_road_name(centroid_lat, centroid_lon, index, original_name):
    # Find closest landmark
    min_dist = float('inf')
    nearest_name = "Campus Center"
    for name, coords in LANDMARKS.items():
        dist = ((coords[0] - centroid_lat) ** 2 + (coords[1] - centroid_lon) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
    # Convert degrees to meters approximately (1 degree latitude ~= 111,000 meters)
    dist_meters = int(min_dist * 111000)
    
    # If original name is a generic placeholder, replace it
    if not original_name or original_name == 'Unnamed Campus Road':
        return f"Road near {nearest_name} ({dist_meters}m)"
    else:
        return f"{original_name} (near {nearest_name}, {dist_meters}m)"

# =====================================================================
def load_assets():
    # Load dataset
    data_path = "data/pavement_dataset.pkl"
    if not os.path.exists(data_path):
        st.error("Pavement dataset file not found. Run Phase 3 simulation first!")
        st.stop()
        
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
        
    # Reconstruct the Dual Graph structure
    adjacency = dataset['adjacency']
    node_list = dataset['node_list']
    coords = dataset['coords']
    geometries = dataset.get('geometries', [])
    gates = dataset.get('gates', [])
    destinations = dataset.get('destinations', [])
    initial_pcis = dataset['x'][-1, :, 0].tolist()
    
    L = nx.DiGraph()
    raw_node_names = dataset.get('node_names', ['Unnamed Campus Road'] * len(node_list))
    simplified_names = []
    
    # Add nodes with saved attributes
    for idx, node in enumerate(node_list):
        length = 150.0 if idx % 3 == 0 else 80.0
        free_flow_time = 15.0
        capacity = 1500 if idx % 3 == 0 else 500
        cbr_dry = 8.0 if idx % 3 == 0 else 5.5
        drainage_quality = 0.85 if idx % 3 == 0 else 0.45
        design_life_msa = 0.15 if idx % 3 == 0 else 0.035
        
        # Calculate subgrade modulus and back-calculate design strains
        mr_sub = phys.calculate_resilient_modulus_subgrade(cbr_dry)
        mr_bit = 3000.0
        eps_t_allow, eps_v_allow = phys.back_calculate_design_strains(design_life_msa, mr_bit, mr_sub)
        
        # Coordinates: coords has shape [N, 2] as [lon, lat]
        centroid_lon = coords[idx, 0]
        centroid_lat = coords[idx, 1]
        
        # Generate simplified name
        orig_name = raw_node_names[idx]
        simp_name = get_simplified_road_name(centroid_lat, centroid_lon, idx, orig_name)
        simplified_names.append(simp_name)
        
        # Geometries start/end
        start_lat, start_lon = geometries[idx][0]
        end_lat, end_lon = geometries[idx][1]
        
        L.add_node(
            node, 
            index=idx,
            centroid_x=centroid_lon,
            centroid_y=centroid_lat,
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            PCI=100.0,
            AADT=1000.0,
            length=length,
            free_flow_time=free_flow_time,
            capacity=capacity,
            cbr_dry=cbr_dry,
            drainage_quality=drainage_quality,
            design_life_msa=design_life_msa,
            mr_bit=mr_bit,
            mr_sub=mr_sub,
            eps_t_allowable=eps_t_allow,
            eps_v_allowable=eps_v_allow,
            name=simp_name
        )
        
    # Reconstruct edges
    num_nodes = len(node_list)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency[i, j] > 0:
                L.add_edge(node_list[i], node_list[j])
                
    # Load trained GNN weights
    gnn_model = PavementPredictorGNN(in_channels=42, hidden_channels=32)
    weights_path = "models/gnn_weights.pt"
    if os.path.exists(weights_path):
        gnn_model.load_state_dict(torch.load(weights_path))
    gnn_model.eval()
    
    # Load static Random Forest baseline
    baseline_path = "models/static_baseline.pkl"
    static_model = None
    if os.path.exists(baseline_path):
        with open(baseline_path, 'rb') as f:
            static_model = pickle.load(f)
            
    return L, gnn_model, static_model, node_list, simplified_names, geometries, gates, destinations, initial_pcis

# Load assets
L, gnn_model, static_model, node_list, node_names_list, geometries_list, gates_list, destinations_list, initial_pcis = load_assets()

# =====================================================================
# 2. SIDEBAR CONTROLS & SESSION STATE INITIALIZATION
# =====================================================================
# Check if a map click occurred in the previous run to programmatically select the road
if "vnit_map" in st.session_state and st.session_state["vnit_map"] is not None:
    map_data = st.session_state["vnit_map"]
    if map_data.get("last_object_clicked"):
        clicked_tooltip = map_data.get("last_object_clicked_tooltip")
        if clicked_tooltip and "Index: " in clicked_tooltip:
            try:
                clicked_idx = int(clicked_tooltip.split("Index: ")[1].split(")")[0].strip())
                road_names_temp = [f"{node_names_list[i]} (Index: {i})" for i in range(len(node_list))]
                if clicked_idx < len(road_names_temp):
                    st.session_state["selected_road_selectbox_val"] = road_names_temp[clicked_idx]
            except Exception as e:
                pass

# Initialize session state for widgets to support programmatic overrides (e.g. Break It button)
if "pci_damage_val" not in st.session_state:
    st.session_state["pci_damage_val"] = 100
if "road_drainage_val" not in st.session_state:
    st.session_state["road_drainage_val"] = 0.5

st.sidebar.header("🕹️ Simulation Controls")

# A. Monsoon Severity Slider
monsoon_severity = st.sidebar.slider(
    "🌧️ Monsoon Severity (Precipitation)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="0.0 = Dry Season. 1.0 = Extreme Monsoon water infiltration."
)

# B. Damage Injection Widget
st.sidebar.subheader("💥 Artificially Damage a Road")
road_names = [f"{node_names_list[idx]} (Index: {idx})" for idx, i in enumerate(node_list)]
selected_road_str = st.sidebar.selectbox(
    "Select Target Road:", 
    road_names,
    key="selected_road_selectbox_val"
)
selected_idx = int(selected_road_str.split("Index: ")[1].replace(")", ""))
selected_road = node_list[selected_idx]

# If the selected road index has changed, sync the sliders to that road's condition
if "last_selected_road_idx" not in st.session_state:
    st.session_state["last_selected_road_idx"] = selected_idx
elif selected_idx != st.session_state["last_selected_road_idx"]:
    st.session_state["pci_damage_val"] = int(initial_pcis[selected_idx])
    st.session_state["road_drainage_val"] = float(L.nodes[selected_road]['drainage_quality'])
    st.session_state["last_selected_road_idx"] = selected_idx

pci_damage = st.sidebar.slider(
    "Set Current PCI Level:",
    min_value=10,
    max_value=100,
    key="pci_damage_val",
    step=5,
    help="""
    Pavement Condition Index (PCI) is a structural and surface quality rating (10 to 100).
    - **100**: Pristine road condition.
    - **50 - 85**: Fair condition with minor cracks/rutting.
    - **< 50**: Poor condition with potholes or structural failure.
    Low PCI increases vehicle roughness delays, forcing commuters onto detour routes.
    """
)

# Dynamic parameter customizers
road_drainage = st.sidebar.slider(
    "Set Target Road Drainage Quality:",
    min_value=0.0,
    max_value=1.0,
    key="road_drainage_val",
    step=0.1,
    help="0.0 = terrible drainage (waterlogging), 1.0 = perfect drainage."
)

traffic_volume_input = st.sidebar.slider(
    "🔄 Base Daily Traffic Volume:",
    min_value=50,
    max_value=1000,
    value=150,
    step=50,
    help="Increase campus traffic volume to see detour failure acceleration."
)

# Callback for "Break Selected Road" button to update session state values before sliders are instantiated
def break_road_callback():
    st.session_state["pci_damage_val"] = 20
    st.session_state["road_drainage_val"] = 0.0

# "Break It" Action Button
st.sidebar.button(
    "💥 Break Selected Road",
    on_click=break_road_callback,
    help="Instantly drop selected road's PCI to 20 and drainage to 0.0"
)

st.sidebar.subheader("🧭 Custom Route Query")
gate_options = {L.nodes[g]['name']: g for g in gates_list if g in L.nodes()}
dest_options = {L.nodes[d]['name']: d for d in destinations_list if d in L.nodes()}

selected_gate_str = st.sidebar.selectbox("Start Gate:", list(gate_options.keys()))
selected_dest_str = st.sidebar.selectbox("End Landmark:", list(dest_options.keys()))
route_strategy = st.sidebar.radio(
    "Routing Policy:", 
    ["⏱️ Fastest (Congestion-Avoidance)", "🛡️ Pavement-Friendly (Structure-Preservation)"]
)
query_gate_node = gate_options[selected_gate_str]
query_dest_node = dest_options[selected_dest_str]

st.sidebar.button("🚀 Run Digital Twin Forecast", help="Forecast runs automatically, but click here to force refresh.")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .reportview-container {
        background-color: #0f1116;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .alert-card {
        background: rgba(255, 165, 0, 0.05);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FFA500;
        margin-bottom: 10px;
        font-size: 13.5px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. INTERACTIVE SIMULATION ROUTING & FORECAST LOOP
# =====================================================================

# A. Calculate Dry Baseline Scenario (unclosed road, dry weather, traffic = 150)
np.random.seed(42)
baseline_traffic_volumes = {}
for node in L.nodes():
    length = L.nodes[node]['length']
    if length > 120.0:
        baseline_traffic_volumes[node] = np.random.uniform(120.0, 320.0)
    else:
        baseline_traffic_volumes[node] = np.random.uniform(25.0, 75.0)

baseline_edge_weights = {}
for u, v in L.edges():
    pci = 100.0  # pristine condition
    free_flow = L.nodes[v]['free_flow_time']
    vol = baseline_traffic_volumes[v]
    cap = L.nodes[v]['capacity']
    weight = phys.calculate_bpr_travel_time(free_flow, vol, cap, pci)
    baseline_edge_weights[(u, v)] = weight

for g in gates_list:
    for d in destinations_list:
        try:
            path = nx.shortest_path(L, source=g, target=d, weight=lambda u, v, dat: baseline_edge_weights.get((u, v), 1.0))
            path_monthly_volume = (150.0 * 30.0) / (len(gates_list) * len(destinations_list))
            for node in path:
                baseline_traffic_volumes[node] += path_monthly_volume
        except nx.NetworkXNoPath:
            continue

# B. Calculate Current Scenario (user inputs)
current_pcis = {node: float(initial_pcis[idx]) for idx, node in enumerate(node_list)}
current_pcis[selected_road] = float(pci_damage)
L.nodes[selected_road]['drainage_quality'] = float(road_drainage)

# Also apply damage to the opposite direction of the selected road segment if it exists
opp_road = None
if isinstance(selected_road, tuple) and len(selected_road) == 2:
    opp_road = (selected_road[1], selected_road[0])
    if opp_road in current_pcis:
        current_pcis[opp_road] = float(pci_damage)
        L.nodes[opp_road]['drainage_quality'] = float(road_drainage)

traffic_volumes = {}
np.random.seed(42)
for node in L.nodes():
    length = L.nodes[node]['length']
    if length > 120.0:
        traffic_volumes[node] = np.random.uniform(120.0, 320.0)
    else:
        traffic_volumes[node] = np.random.uniform(25.0, 75.0)

# Travel times using dynamic BPR
edge_weights = {}
for u, v in L.edges():
    pci = current_pcis[v]
    free_flow = L.nodes[v]['free_flow_time']
    vol = traffic_volumes[v]
    cap = L.nodes[v]['capacity']
    weight = phys.calculate_bpr_travel_time(free_flow, vol, cap, pci)
    edge_weights[(u, v)] = weight

# Route current traffic
for g in gates_list:
    for d in destinations_list:
        try:
            path = nx.shortest_path(L, source=g, target=d, weight=lambda u, v, dat: edge_weights.get((u, v), 1.0))
            path_monthly_volume = (traffic_volume_input * 30.0) / (len(gates_list) * len(destinations_list))
            for node in path:
                traffic_volumes[node] += path_monthly_volume
        except nx.NetworkXNoPath:
            continue

# Assign monthly traffic volume to nodes in DiGraph L
for node in L.nodes():
    L.nodes[node]['monthly_traffic_volume'] = traffic_volumes[node]

# D. Custom Route Calculation
fastest_path = None
friendly_path = None
is_monsoon = monsoon_severity > 0.4
if 'query_gate_node' in locals() and 'query_dest_node' in locals():
    # 1. Calculate Fastest Path (Congestion-Avoidance)
    try:
        fastest_path = nx.shortest_path(L, source=query_gate_node, target=query_dest_node, weight=lambda u, v, dat: edge_weights.get((u, v), 1.0))
    except nx.NetworkXNoPath:
        fastest_path = None
        
    # 2. Calculate Pavement-Friendly Path (Structure-Preservation)
    pavement_friendly_weights = {}
    for u, v in L.edges():
        pci = current_pcis[v]
        drainage = L.nodes[v]['drainage_quality']
        cbr_dry = L.nodes[v]['cbr_dry']
        if is_monsoon:
            m_factor = 0.8 if drainage >= 0.75 else (0.8 - (0.5 * monsoon_severity))
        else:
            m_factor = 1.0
        cbr_eff = cbr_dry * m_factor
        
        pci_penalty = 100.0 - pci
        cbr_penalty = 20.0 / max(cbr_eff, 1.0)
        pavement_friendly_weights[(u, v)] = L.nodes[v]['free_flow_time'] + pci_penalty + cbr_penalty
    
    try:
        friendly_path = nx.shortest_path(L, source=query_gate_node, target=query_dest_node, weight=lambda u, v, dat: pavement_friendly_weights.get((u, v), 1.0))
    except nx.NetworkXNoPath:
        friendly_path = None

custom_path = fastest_path if route_strategy == "⏱️ Fastest (Congestion-Avoidance)" else friendly_path

# C. Assemble normalized input features for current state (Month 0)
is_monsoon = monsoon_severity > 0.4
num_nodes = len(node_list)
x_matrix = np.zeros((num_nodes, 7))

current_mr_sub = {}
current_eps_t = {}
current_eps_v = {}
current_nf = {}
current_nr = {}

for idx, node in enumerate(node_list):
    traffic_vol = traffic_volumes[node]
    cbr_dry = L.nodes[node]['cbr_dry']
    drainage = L.nodes[node]['drainage_quality']
    
    # Calculate effective CBR based on monsoon slider
    if is_monsoon:
        m_factor = 0.8 if drainage >= 0.75 else (0.8 - (0.5 * monsoon_severity))
    else:
        m_factor = 1.0
    cbr_eff = cbr_dry * m_factor
    mr_sub = phys.calculate_resilient_modulus_subgrade(cbr_eff)
    current_mr_sub[node] = mr_sub
    
    # Strain spiking
    eps_t_allow = L.nodes[node]['eps_t_allowable']
    eps_v_allow = L.nodes[node]['eps_v_allowable']
    mod_ratio = L.nodes[node]['mr_sub'] / mr_sub
    eps_t = eps_t_allow * (mod_ratio ** 0.5)
    eps_v = eps_v_allow * (mod_ratio ** 0.8)
    current_eps_t[node] = eps_t
    current_eps_v[node] = eps_v
    
    # Design life limits
    current_nf[node] = phys.calculate_fatigue_life_nf(eps_t, L.nodes[node]['mr_bit'])
    current_nr[node] = phys.calculate_rutting_life_nr(eps_v)
    
    # Normalize features
    x_matrix[idx, 0] = current_pcis[node] / 100.0
    x_matrix[idx, 1] = np.log1p(traffic_vol) / 12.0
    x_matrix[idx, 2] = cbr_eff / 10.0
    x_matrix[idx, 3] = mr_sub / 100.0
    x_matrix[idx, 4] = eps_t * 1e4
    x_matrix[idx, 5] = eps_v * 1e4
    x_matrix[idx, 6] = drainage

# D. Month 12 Prediction (Step 1)
x_history = np.repeat(x_matrix[:, np.newaxis, :], 6, axis=1)  # Shape: [N, 6, 7]
x_flat = x_history.reshape(num_nodes, -1)  # Shape: [N, 42]

tensor_x = torch.tensor(x_flat, dtype=torch.float)
edge_rows, edge_cols = np.where(nx.to_numpy_array(L) > 0)
edge_index = torch.tensor(np.vstack([edge_rows, edge_cols]), dtype=torch.long)
from torch_geometric.data import Data
pyg_data = Data(x=tensor_x, edge_index=edge_index)

with torch.no_grad():
    gnn_preds_12 = gnn_model(pyg_data).numpy().squeeze() * 100.0
    gnn_preds_12 = np.clip(gnn_preds_12, 10.0, 100.0)

if static_model is not None:
    static_preds_12 = static_model.predict(x_flat) * 100.0
    static_preds_12 = np.clip(static_preds_12, 10.0, 100.0)
else:
    static_preds_12 = current_pcis.copy()

# E. Month 24 Prediction (Step 2 Autoregressive)
x_matrix_gnn_24 = x_matrix.copy()
x_matrix_gnn_24[:, 0] = gnn_preds_12 / 100.0
x_history_gnn_24 = np.repeat(x_matrix_gnn_24[:, np.newaxis, :], 6, axis=1)
x_flat_gnn_24 = x_history_gnn_24.reshape(num_nodes, -1)
tensor_x_gnn_24 = torch.tensor(x_flat_gnn_24, dtype=torch.float)
pyg_data_gnn_24 = Data(x=tensor_x_gnn_24, edge_index=edge_index)

with torch.no_grad():
    gnn_preds_24 = gnn_model(pyg_data_gnn_24).numpy().squeeze() * 100.0
    gnn_preds_24 = np.clip(gnn_preds_24, 10.0, 100.0)

x_matrix_static_24 = x_matrix.copy()
x_matrix_static_24[:, 0] = static_preds_12 / 100.0
x_history_static_24 = np.repeat(x_matrix_static_24[:, np.newaxis, :], 6, axis=1)
x_flat_static_24 = x_history_static_24.reshape(num_nodes, -1)

if static_model is not None:
    static_preds_24 = static_model.predict(x_flat_static_24) * 100.0
    static_preds_24 = np.clip(static_preds_24, 10.0, 100.0)
else:
    static_preds_24 = static_preds_12.copy()

# =====================================================================
# 4. DASHBOARD PAGE LAYOUT
# =====================================================================
col1, col2 = st.columns([2, 1])

# Layout Mode / Time selectors above the map inside col1
with col1:
    st.subheader("🗺️ Geographic Digital Twin Map")
    
    col1_ctl1, col1_ctl2 = st.columns(2)
    with col1_ctl1:
        map_mode = st.radio(
            "🗺️ Map Visualization Mode:",
            ["Pavement Condition (PCI)", "Traffic Congestion (Travel Time Delay)"],
            horizontal=True,
            help="""
            - **Pavement Condition (PCI)**: Measures road structural condition (10-100). Lower values indicate surface distress and cracks.
            - **Traffic Congestion (Delay Ratio)**: Calculated as (Current Travel Time / Free-Flow Time). Ratios > 1.2x mean moderate travel delays, and > 2.0x indicate severe traffic gridlock.
            """
        )
    with col1_ctl2:
        forecast_horizon = st.slider(
            "🔮 Forecast Horizon (Months):",
            min_value=0,
            max_value=24,
            value=12,
            step=12,
            help="Forecast horizon. 0 = current, 12 = 12 months forecast, 24 = 24 months forecast."
        )
        
    # Map GNN & Static outputs to variables based on slider selection
    if forecast_horizon == 0:
        predicted_gnn_pcis = {node: float(current_pcis[node]) for node in node_list}
        predicted_static_pcis = {node: float(current_pcis[node]) for node in node_list}
    elif forecast_horizon == 12:
        predicted_gnn_pcis = {node: float(gnn_preds_12[idx]) for idx, node in enumerate(node_list)}
        predicted_static_pcis = {node: float(static_preds_12[idx]) for idx, node in enumerate(node_list)}
    else: # Month 24
        predicted_gnn_pcis = {node: float(gnn_preds_24[idx]) for idx, node in enumerate(node_list)}
        predicted_static_pcis = {node: float(static_preds_24[idx]) for idx, node in enumerate(node_list)}

    # Map Legend based on selected mode
    if map_mode == "Pavement Condition (PCI)":
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-bottom: 15px; font-size: 13px; align-items: center; background: rgba(255,255,255,0.03); padding: 8px 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <b style="color: #9ea4b0; margin-right: 5px; cursor: help;" title="Pavement Condition Index (PCI) is a structural and surface quality rating from 10 (failed) to 100 (perfect). GNN forecasts how PCI will decay over the selected months based on subgrade moisture and detour traffic loads.">Pavement Condition (PCI) 🛈</b>
            <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="height: 10px; width: 10px; background-color: #2ecc71; border-radius: 50%; display: inline-block;"></span><b>Good Condition</b> (PCI &gt; 85)</span>
            <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="height: 10px; width: 10px; background-color: #f39c12; border-radius: 50%; display: inline-block;"></span><b>Fair Condition</b> (PCI 50 - 85)</span>
            <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="height: 10px; width: 10px; background-color: #e74c3c; border-radius: 50%; display: inline-block;"></span><b>Poor Condition</b> (PCI &lt; 50)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-bottom: 15px; font-size: 13px; align-items: center; background: rgba(255,255,255,0.03); padding: 8px 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <b style="color: #9ea4b0; margin-right: 5px; cursor: help;" title="Delay Ratio is calculated as (Current Travel Time / Free-Flow Time). Travel Time includes baseline congestion delay plus a pavement roughness penalty (0.05 seconds per point of PCI drop).">Traffic Congestion (Delay Ratio) 🛈</b>
            <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="height: 10px; width: 10px; background-color: #2ecc71; border-radius: 50%; display: inline-block;"></span><b>No Congestion</b> (Delay &le; 1.2x)</span>
            <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="height: 10px; width: 10px; background-color: #f39c12; border-radius: 50%; display: inline-block;"></span><b>Moderate Delay</b> (Delay 1.2x - 2.0x)</span>
            <span style="display: inline-flex; align-items: center; gap: 5px;"><span style="height: 10px; width: 10px; background-color: #e74c3c; border-radius: 50%; display: inline-block;"></span><b>Severe Congestion</b> (Delay &gt; 2.0x)</span>
        </div>
        """, unsafe_allow_html=True)

    # Debug print statements
    print(f"DEBUG: Selected Road = {selected_road}")
    print(f"DEBUG: Input PCI Damage = {pci_damage}")
    print(f"DEBUG: GNN Pred for Selected Road = {predicted_gnn_pcis[selected_road]}")

    # Initialize Folium Map centered on selected road
    if 'selected_road' in locals() and selected_road in L.nodes():
        vnit_lat = L.nodes[selected_road]['centroid_y']
        vnit_lon = L.nodes[selected_road]['centroid_x']
        zoom_start = 17
        st.info(f"📍 Map auto-centered on selected target road: **{L.nodes[selected_road]['name']}**")
    else:
        vnit_lat, vnit_lon = 21.1265, 79.0504
        zoom_start = 15
        
    m = folium.Map(location=[vnit_lat, vnit_lon], zoom_start=zoom_start, tiles="cartodbpositron")
    
    # Draw roads
    for idx, node in enumerate(node_list):
        lat = L.nodes[node]['centroid_y']
        lon = L.nodes[node]['centroid_x']
        name = L.nodes[node]['name']
        
        # Color encoding logic
        if map_mode == "Pavement Condition (PCI)":
            pci_val = predicted_gnn_pcis[node]
            if pci_val > 85.0:
                color = "green"
            elif pci_val > 50.0:
                color = "orange"
            else:
                color = "red"
        else: # Traffic Congestion Mode
            vol = traffic_volumes[node]
            cap = L.nodes[node]['capacity']
            pci = current_pcis[node]
            free_flow = L.nodes[node]['free_flow_time']
            travel_time = phys.calculate_bpr_travel_time(free_flow, vol, cap, pci)
            delay_ratio = travel_time / free_flow
            
            if delay_ratio <= 1.2:
                color = "green"
            elif delay_ratio <= 2.0:
                color = "orange"
            else:
                color = "red"
            
        cbr_dry = L.nodes[node]['cbr_dry']
        drainage = L.nodes[node]['drainage_quality']
        mr_sub = current_mr_sub[node]
        eps_t = current_eps_t[node]
        eps_v = current_eps_v[node]
        eps_t_allow = L.nodes[node]['eps_t_allowable']
        eps_v_allow = L.nodes[node]['eps_v_allowable']
        nf = current_nf[node]
        nr = current_nr[node]
        
        # Draw road node marker
        popup_info = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 260px;">
            <b style="color: #1E90FF; font-size: 13px;">{name}</b><br>
            <b>OSM Directions Link:</b> <a href="https://www.openstreetmap.org/directions?from=&to={lat}%2C{lon}" target="_blank">View on OSM</a><br>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Current PCI:</b></td><td>{current_pcis[node]:.1f}</td></tr>
                <tr><td><b>Predicted PCI ({forecast_horizon}m):</b></td><td><span style="color: {color if map_mode == "Pavement Condition (PCI)" else 'black'}; font-weight: bold;">{predicted_gnn_pcis[node]:.1f}</span></td></tr>
                <tr><td><b>Traffic Volume:</b></td><td>{traffic_volumes[node]:.1f} veh/mo</td></tr>
                <tr><td><b>Drainage Quality:</b></td><td>{drainage:.2f} ({'Good' if drainage >= 0.75 else 'Poor'})</td></tr>
                <tr><td><b>CBR (Effective):</b></td><td>{cbr_dry * (mr_sub / L.nodes[node]['mr_sub']):.2f}% (Dry: {cbr_dry:.1f}%)</td></tr>
                <tr><td><b>Subgrade Modulus:</b></td><td>{mr_sub:.1f} MPa</td></tr>
            </table>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
            <b style="color: #555;">Structural Strains & Allowable (IRC:37):</b><br>
            - Tensile (ε_t): {eps_t * 1e6:.1f} με (Allow: {eps_t_allow * 1e6:.1f} με)<br>
            - Compressive (ε_v): {eps_v * 1e6:.1f} με (Allow: {eps_v_allow * 1e6:.1f} με)<br>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
            <b style="color: #555;">Pavement Fatigue/Rutting Design Limits:</b><br>
            - Fatigue Life (N_f): {nf / 1e6:.4f} MSA<br>
            - Rutting Life (N_r): {nr / 1e6:.4f} MSA<br>
        </div>
        """
        
        start_lat = L.nodes[node]['start_lat']
        start_lon = L.nodes[node]['start_lon']
        end_lat = L.nodes[node]['end_lat']
        end_lon = L.nodes[node]['end_lon']
        
        # Draw target road with cyan highlight halo but display its actual state color on the main segment line
        is_selected = (node == selected_road) or (node == opp_road)
        is_in_friendly_path = (friendly_path is not None) and (node in friendly_path)
        is_in_fastest_path = (fastest_path is not None) and (node in fastest_path)
        
        # Draw Pavement-Friendly path (thick blue line)
        if is_in_friendly_path:
            folium.PolyLine(
                locations=[(start_lat, start_lon), (end_lat, end_lon)],
                color="#0070FF",  # Blue path highlight
                weight=10,
                opacity=0.7,
                tooltip=f"Pavement-Friendly Route: {name} (Index: {idx})"
            ).add_to(m)
            
        # Draw Fastest path (thinner black line overlay)
        if is_in_fastest_path:
            folium.PolyLine(
                locations=[(start_lat, start_lon), (end_lat, end_lon)],
                color="#000000",  # Black path highlight
                weight=5,
                opacity=0.9,
                tooltip=f"Fastest Route: {name} (Index: {idx})"
            ).add_to(m)

        if is_selected:
            folium.PolyLine(
                locations=[(start_lat, start_lon), (end_lat, end_lon)],
                color="#00FFFF",  # Cyan glow halo
                weight=12,
                opacity=0.6,
                tooltip=f"{name} (Index: {idx})"
            ).add_to(m)
            
            folium.PolyLine(
                locations=[(start_lat, start_lon), (end_lat, end_lon)],
                color=color,  # Segment color represents state
                weight=6,
                opacity=0.9,
                popup=folium.Popup(popup_info, max_width=300),
                tooltip=f"{name} (Index: {idx})"
            ).add_to(m)
        else:
            folium.PolyLine(
                locations=[(start_lat, start_lon), (end_lat, end_lon)],
                color=color,
                weight=5,
                opacity=0.8,
                popup=folium.Popup(popup_info, max_width=300),
                tooltip=f"{name} (Index: {idx})"
            ).add_to(m)
            
        # Draw centroid markers
        if is_selected:
            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#1E90FF",
                weight=3,
                fill=True,
                fill_color="#00FFFF",
                fill_opacity=0.8,
                popup=folium.Popup(popup_info, max_width=300),
                tooltip=f"{name} (Index: {idx})"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_info, max_width=300),
                tooltip=f"{name} (Index: {idx})"
            ).add_to(m)
            
    # Draw custom path start and end pins
    if friendly_path or fastest_path:
        folium.Marker(
            location=[L.nodes[query_gate_node]['centroid_y'], L.nodes[query_gate_node]['centroid_x']],
            popup="🚦 ROUTE STARTING POINT",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)
        
        folium.Marker(
            location=[L.nodes[query_dest_node]['centroid_y'], L.nodes[query_dest_node]['centroid_x']],
            popup="🏁 ROUTE DESTINATION",
            icon=folium.Icon(color="red", icon="flag", prefix="fa")
        ).add_to(m)

    # Use st_folium instead of folium_static to capture map click interactions, limiting returned objects to prevent zoom/pan reruns
    st_folium(
        m, 
        width=750, 
        height=500, 
        key="vnit_map", 
        returned_objects=["last_object_clicked", "last_object_clicked_tooltip"]
    )

with col2:
    st.subheader("📊 Cascade Failure Analysis")
    
    gnn_failures = sum(1 for n in L.nodes() if predicted_gnn_pcis[n] < 50.0)
    static_failures = sum(1 for n in L.nodes() if predicted_static_pcis[n] < 50.0)
    
    # 1. Metric display
    st.metric(
        label="🚨 GNN Predicted Vulnerable Roads",
        value=f"{gnn_failures} segments",
        delta=f"{gnn_failures - static_failures} vs. Static model",
        delta_color="inverse"
    )
    
    # Custom Route Statistics Card
    if custom_path:
        total_path_time = sum(phys.calculate_bpr_travel_time(
            L.nodes[n]['free_flow_time'], 
            traffic_volumes[n], 
            L.nodes[n]['capacity'], 
            current_pcis[n]
        ) for n in custom_path)
        
        avg_pci = np.mean([predicted_gnn_pcis[n] for n in custom_path])
        avg_cbr = np.mean([L.nodes[n]['cbr_dry'] * (current_mr_sub[n] / L.nodes[n]['mr_sub']) for n in custom_path])
        
        if avg_pci > 85.0:
            status_html = "<span style='color: #2ecc71; font-weight: bold;'>🟢 Excellent & Smooth</span>"
        elif avg_pci > 50.0:
            status_html = "<span style='color: #f39c12; font-weight: bold;'>🟡 Fair / Some Roughness</span>"
        else:
            status_html = "<span style='color: #e74c3c; font-weight: bold;'>🔴 Poor / High Wear</span>"
            
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #0070FF; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #0070FF;">🧭 Path: {selected_gate_str} ➔ {selected_dest_str}</h4>
            <table style="width: 100%; font-size: 13.5px; border-collapse: collapse;">
                <tr><td><b>Strategy:</b></td><td>{route_strategy.split(' ')[1]}</td></tr>
                <tr><td><b>Total Travel Time:</b></td><td>{total_path_time:.1f} seconds</td></tr>
                <tr><td><b>Average PCI:</b></td><td>{avg_pci:.1f}</td></tr>
                <tr><td><b>Effective CBR:</b></td><td>{avg_cbr:.2f}%</td></tr>
                <tr><td><b>Ride Quality:</b></td><td>{status_html}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #0070FF; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #0070FF;">🧭 Custom Route Analysis</h4>
            <p style="font-size: 13.5px; margin: 0; color: #888;">No path found between selected gate and landmark.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Accelerated Decay Alerts List
    st.markdown("### ⚠️ Accelerated Pavement Decay Alerts")
    alerts_list = []
    VDF_composite = 0.1986
    
    for idx, node in enumerate(node_list):
        if node == selected_road or node == opp_road:
            continue
            
        # Baseline remaining life
        base_vol = baseline_traffic_volumes[node]
        base_annual_axles = base_vol * 12.0 * VDF_composite
        base_nf = phys.calculate_fatigue_life_nf(L.nodes[node]['eps_t_allowable'], L.nodes[node]['mr_bit'])
        base_nr = phys.calculate_rutting_life_nr(L.nodes[node]['eps_v_allowable'])
        base_life = min(base_nf, base_nr) / max(base_annual_axles, 1.0)
        
        # Current remaining life
        curr_vol = traffic_volumes[node]
        curr_annual_axles = curr_vol * 12.0 * VDF_composite
        curr_life = min(current_nf[node], current_nr[node]) / max(curr_annual_axles, 1.0)
        
        if curr_life < base_life and curr_life < 15.0:
            reduction = base_life - curr_life
            reduction_pct = (reduction / base_life) * 100.0
            if reduction_pct > 15.0:  # Only report significant decay drops
                alerts_list.append({
                    "name": L.nodes[node]['name'],
                    "idx": L.nodes[node]['index'],
                    "base_life": base_life,
                    "curr_life": curr_life,
                    "reduction": reduction
                })
                
    # Sort alerts by remaining life reduction amount (highest reduction first)
    alerts_list = sorted(alerts_list, key=lambda x: x["reduction"], reverse=True)
    
    if alerts_list:
        # Show top 3 alerts
        for alert in alerts_list[:3]:
            name = alert["name"]
            idx = alert["idx"]
            b_life = alert["base_life"]
            c_life = alert["curr_life"]
            
            if c_life < 1.0:
                c_life_str = f"{c_life * 12.0:.0f} months"
            else:
                c_life_str = f"{c_life:.1f} years"
                
            st.markdown(f"""
            <div class="alert-card">
                <b>⚠️ {name}</b> (Index: {idx})<br>
                Remaining Design Life reduced from <b>{b_life:.1f} years</b> to <b>{c_life_str}</b> due to detour loading / subgrade moisture stress!
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No critical accelerated pavement decay alerts detected.")
    
    # 3. Model comparison graph on top detour route
    st.markdown("### 📈 ST-GNN vs. Static Model Decay Plot")
    # Identify detour segment with largest traffic volume increase relative to its baseline
    max_increase = -1.0
    detour_node = None
    for node in L.nodes():
        if node != selected_road and node != opp_road:
            increase = traffic_volumes[node] - baseline_traffic_volumes[node]
            if increase > max_increase:
                max_increase = increase
                detour_node = node
                
    if detour_node:
        detour_idx = L.nodes[detour_node]['index']
        detour_name = L.nodes[detour_node]['name']
        
        st.markdown(f"Detour road segment monitored:  \n**{detour_name} (Index: {detour_idx})**")
        
        # Plot predictions over 0, 12, and 24 months
        chart_df = pd.DataFrame({
            "Timeline": [0, 12, 24],
            "ST-GNN Model": [
                float(current_pcis[detour_node]),
                float(gnn_preds_12[detour_idx]),
                float(gnn_preds_24[detour_idx])
            ],
            "Static Model": [
                float(current_pcis[detour_node]),
                float(static_preds_12[detour_idx]),
                float(static_preds_24[detour_idx])
            ]
        })
        st.line_chart(chart_df.set_index("Timeline"))
    
    # 4. Detour suspects detail table
    st.subheader("📋 Top Detour Failure Suspects")
    suspect_data = []
    for node in L.nodes():
        if node != selected_road and node != opp_road and predicted_gnn_pcis[node] < 85.0:
            suspect_data.append({
                "Road Name": L.nodes[node]['name'],
                "Index": L.nodes[node]['index'],
                "GNN Pred (PCI)": f"{predicted_gnn_pcis[node]:.1f}",
                "Static Pred (PCI)": f"{predicted_static_pcis[node]:.1f}",
                "Traffic Volume": f"{traffic_volumes[node]:.1f}"
            })
            
    if suspect_data:
        # Sort suspects by GNN Pred (PCI) in ascending order (lowest first)
        suspect_data = sorted(suspect_data, key=lambda x: float(x["GNN Pred (PCI)"]))
        df_suspects = pd.DataFrame(suspect_data).head(5)
        st.table(df_suspects.set_index("Road Name"))
    else:
        st.write("No detour failures predicted yet. Try dropping the target road PCI or cranking up the monsoon severity!")
