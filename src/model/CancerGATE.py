import torch
from torch_geometric.nn import GAE, VGAE, GCNConv, GATConv

class GCNEncoder(torch.nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        self.conv1 = GATConv(dim_list[0], dim_list[1], 8)
        self.conv2 = GATConv(dim_list[0], dim_list[2], 8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(GCNEncoder(13, 64, 32))