import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# =====================================================================
# SPATIO-TEMPORAL GNN ARCHITECTURE
# =====================================================================
# This model uses Graph Attention layers (GAT) to propagate distress
# and traffic load patterns spatially across the Nagpur road network.
# It includes a residual skip connection to preserve node-specific features
# and prevent over-smoothing.
# =====================================================================

class PavementPredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(PavementPredictorGNN, self).__init__()
        
        # 1. Graph Convolution pathways
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        
        # 2. Residual skip connection to prevent over-smoothing
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # 3. Dense regression layers
        self.fc1 = nn.Linear(hidden_channels, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, data):
        """
        Executes GNN forward propagation:
          - data.x: Node features [N, 42]
          - data.edge_index: Graph adjacency connections [2, E]
        """
        x, edge_index = data.x, data.edge_index
        
        # GNN feature aggregation
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        # Add residual connection from input to preserve node identity
        skip = F.relu(self.input_proj(x))
        out = h + skip
        
        # Regression projection
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        
        # Sigmoid scaling to smoothly map output to [0.1, 1.0] without killing gradients
        return 0.1 + 0.9 * torch.sigmoid(out)

if __name__ == "__main__":
    # Test model initialization
    model = PavementPredictorGNN(in_channels=42)
    print("GNN Model successfully compiled!")
    print(model)
