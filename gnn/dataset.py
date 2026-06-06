import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data

def load_pavement_dataset(data_path="data/pavement_dataset.pkl", history_window=6, lead_time=12):
    """
    Loads the simulated time-series dataset and formats it for PyTorch Geometric GNNs.
    
    Inputs:
      - history_window: 6 months of history to look back.
      - lead_time: 12 months ahead to predict.
      
    Returns:
      - data_list: List of PyG Data objects.
      - num_features: Total number of features per node (6 months * 7 features = 42).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}. Run Phase 3 first!")
        
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
        
    # x: [Time, Nodes, Features] -> [120, 256, 7]
    raw_x = dataset['x'].copy()
    
    # Feature Normalization (to prevent gradient saturation / dead ReLUs)
    raw_x[:, :, 0] /= 100.0  # PCI: [10.0, 100.0] -> [0.1, 1.0]
    raw_x[:, :, 1] = np.log1p(raw_x[:, :, 1]) / 12.0  # Traffic: Log-scaled -> [0.0, 0.9]
    raw_x[:, :, 2] /= 10.0  # CBR: [0.0, 10.0] -> [0.0, 1.0]
    raw_x[:, :, 3] /= 100.0  # Subgrade Modulus: [0.0, 100.0] -> [0.0, 1.0]
    raw_x[:, :, 4] *= 1e4  # Scale strain_t to [0.1, 1.0]
    raw_x[:, :, 5] *= 1e4  # Scale strain_v to [0.1, 1.0]
    # Feature 6 (drainage) is already [0.0, 1.0]
    
    adjacency = dataset['adjacency']
    coords = dataset['coords']
    
    num_timesteps, num_nodes, num_features = raw_x.shape
    
    # Convert adjacency matrix [N, N] to PyG COO coordinate list [2, E]
    # np.where finds non-zero links representing intersection connections
    edge_rows, edge_cols = np.where(adjacency > 0)
    edge_index = torch.tensor(np.vstack([edge_rows, edge_cols]), dtype=torch.long)
    
    data_list = []
    
    # We construct sliding window sequences:
    # Example: If m = 18, we take history [12, 13, 14, 15, 16, 17] (6 steps) to predict target [30] (12 steps ahead)
    # Range limit: starts at history_window, ends at total_steps - lead_time
    start_idx = history_window
    end_idx = num_timesteps - lead_time
    
    for t in range(start_idx, end_idx):
        # 1. Gather historical feature window of size [history_window, Nodes, Features]
        # Then, reshape to [Nodes, history_window * Features] to create a single static node feature vector
        history_slice = raw_x[t - history_window : t, :, :]  # Shape: [6, N, 7]
        
        # Transpose to [Nodes, history_window, Features] and flatten the temporal dimension
        history_slice = np.transpose(history_slice, (1, 0, 2))  # Shape: [N, 6, 7]
        node_features = history_slice.reshape(num_nodes, -1)  # Shape: [N, 42]
        
        # 2. Gather target PCI at t + lead_time
        # Shape: [Nodes, 1]
        target_pci = raw_x[t + lead_time, :, 0:1]
        
        # 3. Convert to PyTorch tensors
        tensor_x = torch.tensor(node_features, dtype=torch.float)
        tensor_y = torch.tensor(target_pci, dtype=torch.float)
        
        # 4. Construct the PyTorch Geometric Data object
        data_step = Data(
            x=tensor_x,
            edge_index=edge_index,
            y=tensor_y,
            pos=torch.tensor(coords, dtype=torch.float)  # Geographic coordinates
        )
        data_list.append(data_step)
        
    print(f"Dataset Formatted: Created {len(data_list)} spatial graph samples.")
    # Return list of graphs, and input size (history_window * 7 = 42 features)
    return data_list, history_window * num_features

if __name__ == "__main__":
    # Test dataset reader
    data, features = load_pavement_dataset()
    print(f"First Graph Sample:")
    print(f"  - Node features (x) shape: {data[0].x.shape}")
    print(f"  - Target labels (y) shape: {data[0].y.shape}")
    print(f"  - Adjacency links (edge_index) shape: {data[0].edge_index.shape}")
