import torch
from torch.nn import Linear
import torch.nn.functional as F

############################## PyTorch Implementation (index_add) ##############################
# Graph Convolution Layer Implementation
class GraphConvLayer_pyT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adjacency = torch.zeros(num_nodes, num_nodes, device=x.device)

        # Fill the adjacency matrix with edge connections
        adjacency[edge_index[0], edge_index[1]] = 1
        
        # Step 2: Add self-loops
        adjacency += torch.eye(num_nodes)

        # Step 3: Compute degree matrix
        degree = adjacency.sum(dim=1, keepdim=True)
        # print(degree)

        # Step 4: Normalize the adjacency matrix
        norm = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
        norm = norm * norm.t()
        norm = norm * adjacency

        ### Actual model
        x = self.linear(x)
        out = torch.matmul(norm, x)
        return out

# Define the GCN model
class GCN_pyT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.gcn = GraphConvLayer_pyT(num_features, 64)
        self.out = Linear(64, num_classes)

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return z
        # return F.log_softmax(z, dim=1)