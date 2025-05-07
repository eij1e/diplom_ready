import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

# --- GCN Encoder
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

# --- GIN Encoder
class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

# --- Graph Similarity Classifier
class GraphSimilarityModel(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        z1 = self.encoder(x1, edge_index1, batch1)
        z2 = self.encoder(x2, edge_index2, batch2)
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        return self.classifier(combined).squeeze(-1)
