import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import simulator.network as net
import simulator.physics as phys

# =====================================================================
# PHASE 3: PAVEMENT NETWORK SIMULATOR & DATASET GENERATOR
# =====================================================================
# This script orchestrates a 10-year rolling simulation of the campus
# pavement grid, combining traffic routing, axle loads, and weather.
# =====================================================================

class PavementNetworkSimulator:
    def __init__(self, place_name="Visvesvaraya National Institute of Technology, Nagpur, India"):
        # 1. Download and transform the street graph
        self.G = net.download_campus_network(place_name)
        self.L = net.transform_to_dual_graph(self.G)
        
        # 2. Define campus traffic sources and sinks spread across the campus network
        # We select random nodes from different parts of the graph for diverse routing paths
        all_nodes = list(self.L.nodes())
        np.random.seed(42)
        # Select 15 origins and 45 destinations representing gates, offices, quarters, and hostels
        gate_indices = np.random.choice(len(all_nodes), size=min(15, len(all_nodes)), replace=False)
        dest_indices = np.random.choice(len(all_nodes), size=min(45, len(all_nodes)), replace=False)
        self.gates = [all_nodes[idx] for idx in gate_indices]
        self.destinations = [all_nodes[idx] for idx in dest_indices]
        
        # 3. Initialize physical and design characteristics for every road segment (dual node)
        print("Initializing structural pavement database...")
        np.random.seed(42)  # For reproducible synthetic data
        
        for node in self.L.nodes():
            length = self.L.nodes[node]['length']
            
            # Categorize road: primary loop roads vs. secondary residential access
            if length > 120.0:
                # Major Campus Road (Thin flexible pavement) - Balanced scale
                self.L.nodes[node]['design_life_msa'] = np.random.uniform(0.05, 0.1)
                self.L.nodes[node]['cbr_dry'] = np.random.uniform(7.0, 9.0)  # Dry CBR percentage
                self.L.nodes[node]['drainage_quality'] = np.random.uniform(0.75, 1.0)  # Good drainage
                self.L.nodes[node]['capacity'] = 1500  # Vehicles per hour capacity
                self.L.nodes[node]['free_flow_time'] = (length / (30.0 * 1000 / 3600))  # 30 km/h speed limit
            else:
                # Minor / Residential Road (Very thin access pavement) - Balanced scale
                self.L.nodes[node]['design_life_msa'] = np.random.uniform(0.01, 0.02)
                self.L.nodes[node]['cbr_dry'] = np.random.uniform(5.0, 7.0)  # Lower Dry CBR
                self.L.nodes[node]['drainage_quality'] = np.random.uniform(0.2, 0.6)  # Poor drainage
                self.L.nodes[node]['capacity'] = 500   # Lower capacity
                self.L.nodes[node]['free_flow_time'] = (length / (20.0 * 1000 / 3600))  # 20 km/h speed limit

            # Establish initial structural parameters
            self.L.nodes[node]['CBR'] = self.L.nodes[node]['cbr_dry']
            self.L.nodes[node]['PCI'] = 100.0  # Start fully healthy
            self.L.nodes[node]['cumulative_damage'] = 0.0
            
            # Modulus of Bituminous layer at design temp 35°C (Standard VG-30 bitumen: 3000 MPa)
            self.L.nodes[node]['mr_bit'] = 3000.0 
            
            # Initial dry subgrade resilient modulus
            self.L.nodes[node]['mr_sub'] = phys.calculate_resilient_modulus_subgrade(self.L.nodes[node]['cbr_dry'])
            
            # Back-calculate design structural strains matching the road's classification
            eps_t, eps_v = phys.back_calculate_design_strains(
                self.L.nodes[node]['design_life_msa'],
                self.L.nodes[node]['mr_bit'],
                self.L.nodes[node]['mr_sub']
            )
            self.L.nodes[node]['eps_t_allowable'] = eps_t
            self.L.nodes[node]['eps_v_allowable'] = eps_v
            
            # Current strains start equal to allowable strains (perfectly designed pavement)
            self.L.nodes[node]['eps_t'] = eps_t
            self.L.nodes[node]['eps_v'] = eps_v
            
            # Current traffic starts at zero
            self.L.nodes[node]['monthly_traffic_volume'] = 0.0

    def route_traffic(self, base_daily_commercial_volume=150):
        """
        Routes campus commercial traffic using BPR-weighted shortest path routing on the dual graph.
        Simulates drivers choosing paths based on travel time and pavement roughness.
        """
        # Initialize each road with a baseline local traffic volume so every road degrades naturally
        for node in self.L.nodes():
            length = self.L.nodes[node]['length']
            if length > 120.0:
                self.L.nodes[node]['monthly_traffic_volume'] = np.random.uniform(120.0, 320.0)
            else:
                self.L.nodes[node]['monthly_traffic_volume'] = np.random.uniform(25.0, 75.0)
            
        # Calculate dynamic travel impedance for every road (dual node)
        edge_weights = {}
        for u, v in self.L.edges():
            # In the line graph, nodes are road segments.
            # Travel impedance is determined by the destination road segment's current state.
            pci = self.L.nodes[v]['PCI']
            free_flow = self.L.nodes[v]['free_flow_time']
            volume = self.L.nodes[v]['monthly_traffic_volume']
            capacity = self.L.nodes[v]['capacity']
            
            # Calculate weight using BPR impedance formula
            weight = phys.calculate_bpr_travel_time(free_flow, volume, capacity, pci)
            edge_weights[(u, v)] = weight

        # Distribute traffic from Gates (Origins) to Departments (Sinks)
        for origin in self.gates:
            for dest in self.destinations:
                try:
                    # Find shortest path based on current travel time weights
                    path = nx.shortest_path(self.L, source=origin, target=dest, weight=lambda u, v, d: edge_weights.get((u, v), 1.0))
                    
                    # Assume a portion of the daily commercial vehicles travels this path
                    path_daily_volume = base_daily_commercial_volume / (len(self.gates) * len(self.destinations))
                    path_monthly_volume = path_daily_volume * 30.0  # Monthly traffic
                    
                    for node in path:
                        self.L.nodes[node]['monthly_traffic_volume'] += path_monthly_volume
                except nx.NetworkXNoPath:
                    continue

    def simulate_month(self, month_index):
        """
        Runs a single month of traffic loading, monsoon moisture cycles, and structural decay.
        """
        # 1. Determine if it is a monsoon month in Nagpur (June, July, August, September)
        # In a 12-month calendar (0-indexed): 5 = June, 6 = July, 7 = August, 8 = September
        month_of_year = month_index % 12
        is_monsoon = month_of_year in [5, 6, 7, 8]
        
        # 2. Run the Traffic Assignment Loop (use balanced baseline commercial traffic)
        self.route_traffic(base_daily_commercial_volume=200)
        
        # 3. Calculate VDF and damage accumulation for each road segment
        # Traffic split with balanced wheel loads to represent campus service vehicles
        VDF_composite = 0.85 * phys.calculate_axle_equivalence('single_wheel', 20.0) + \
                        0.15 * phys.calculate_axle_equivalence('single_dual', 85.0)
                        
        for node in self.L.nodes():
            # Get road parameters
            cbr_dry = self.L.nodes[node]['cbr_dry']
            drainage = self.L.nodes[node]['drainage_quality']
            mr_bit = self.L.nodes[node]['mr_bit']
            eps_t_allow = self.L.nodes[node]['eps_t_allowable']
            eps_v_allow = self.L.nodes[node]['eps_v_allowable']
            
            # Apply Monsoon Weakening
            cbr_effective = phys.simulate_monsoon_weakening(cbr_dry, drainage, is_monsoon)
            self.L.nodes[node]['CBR'] = cbr_effective
            
            # Recalculate subgrade resilient modulus
            mr_sub = phys.calculate_resilient_modulus_subgrade(cbr_effective)
            self.L.nodes[node]['mr_sub'] = mr_sub
            
            # Strain Spiking (Hooke's Law approximation due to weakened subgrade)
            # If the subgrade modulus plummets, the operating strains spike
            modulus_ratio = self.L.nodes[node]['mr_sub'] / mr_sub
            self.L.nodes[node]['eps_t'] = eps_t_allow * (modulus_ratio ** 0.5)
            self.L.nodes[node]['eps_v'] = eps_v_allow * (modulus_ratio ** 0.8)
            
            # Calculate fatigue life Nf and rutting life Nr under current strains
            nf = phys.calculate_fatigue_life_nf(self.L.nodes[node]['eps_t'], mr_bit)
            nr = phys.calculate_rutting_life_nr(self.L.nodes[node]['eps_v'])
            
            # Calculate standard axle load repetitions this month
            monthly_commercial_count = self.L.nodes[node]['monthly_traffic_volume']
            monthly_standard_axles = monthly_commercial_count * VDF_composite
            
            # Accumulate mechanical damage ratio (Miner's Rule)
            # Damage = max(Fatigue damage, Rutting damage)
            fatigue_damage_increment = monthly_standard_axles / max(nf, 1.0)
            rutting_damage_increment = monthly_standard_axles / max(nr, 1.0)
            monthly_damage = max(fatigue_damage_increment, rutting_damage_increment)
            
            # Scale up damage for poor drainage during monsoon (representing subgrade stripping/erosion)
            if is_monsoon and drainage < 0.75:
                monthly_damage *= 2.0
                
            self.L.nodes[node]['cumulative_damage'] += monthly_damage
            
            # Update Pavement Condition Index (PCI)
            # PCI drops from 100 to 10 as cumulative damage approaches 1.0
            pci_decay = 100.0 - 90.0 * self.L.nodes[node]['cumulative_damage']
            self.L.nodes[node]['PCI'] = max(pci_decay, 10.0)  # Cap at minimum PCI of 10

    def run_simulation(self, years=10, output_dir="data"):
        """
        Runs the 10-year rolling simulation and saves weekly/monthly dataset matrices.
        """
        os.makedirs(output_dir, exist_ok=True)
        total_months = years * 12
        print(f"Starting {years}-year ({total_months} months) pavement network simulation...")
        
        # We will collect node feature matrices for each month
        # Features: [PCI, TrafficVolume, CBR, ModulusSubgrade, StrainT, StrainV, Drainage]
        num_nodes = len(self.L.nodes())
        num_features = 7
        
        # Dimensions: [Time, Nodes, Features]
        dataset_x = np.zeros((total_months, num_nodes, num_features))
        
        for m in range(total_months):
            self.simulate_month(m)
            
            # Extract and store features for this month
            for i, node in enumerate(self.L.nodes()):
                dataset_x[m, i, 0] = self.L.nodes[node]['PCI']
                dataset_x[m, i, 1] = self.L.nodes[node]['monthly_traffic_volume']
                dataset_x[m, i, 2] = self.L.nodes[node]['CBR']
                dataset_x[m, i, 3] = self.L.nodes[node]['mr_sub']
                dataset_x[m, i, 4] = self.L.nodes[node]['eps_t']
                dataset_x[m, i, 5] = self.L.nodes[node]['eps_v']
                dataset_x[m, i, 6] = self.L.nodes[node]['drainage_quality']
                
            if (m + 1) % 12 == 0:
                avg_pci = np.mean(dataset_x[m, :, 0])
                print(f"  - Year {(m+1)//12} Completed. Network Average PCI: {avg_pci:.2f}")

        # Extract static dual graph adjacency matrix
        adj_matrix = nx.to_numpy_array(self.L)
        
        # Map node coordinates (for geographic mapping in degrees)
        node_coords = np.array([[self.L.nodes[n]['centroid_lon'], self.L.nodes[n]['centroid_lat']] for n in self.L.nodes()])
        
        # Map node labels (tuple labels to list indices)
        node_list = list(self.L.nodes())
        
        # Compile list of human-readable street names
        node_names = [self.L.nodes[n].get('name', 'Unnamed Campus Road') for n in self.L.nodes()]
        
        # Compile road line geometries (WGS84 endpoints)
        geometries = []
        for n in self.L.nodes():
            geometries.append([
                (self.L.nodes[n]['start_lat'], self.L.nodes[n]['start_lon']),
                (self.L.nodes[n]['end_lat'], self.L.nodes[n]['end_lon'])
            ])
            
        dataset = {
            'x': dataset_x,             # Pavement state time-series [120, 256, 7]
            'adjacency': adj_matrix,    # Network connection matrix [256, 256]
            'coords': node_coords,      # Coordinates [256, 2]
            'node_list': node_list,      # Original node labels mapping
            'node_names': node_names,    # Street names
            'geometries': geometries,    # Line string endpoint coordinates
            'gates': self.gates,         # Origin nodes
            'destinations': self.destinations # Destination nodes
        }
        
        data_path = os.path.join(output_dir, "pavement_dataset.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(dataset, f)
            
        print(f"Simulation Finished! Dataset saved to: {data_path}")
        return dataset

if __name__ == "__main__":
    # Test script: Run simulation
    sim = PavementNetworkSimulator()
    sim.run_simulation(years=10)
