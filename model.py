import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from torch.nn import Sequential as Seq, Linear, ReLU

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=1))

class GLaRE(nn.Module):
    def __init__(self, num_features=19, hidden_dim=64, num_classes=8, num_regions=8):
        super().__init__()
        self.num_regions = num_regions
        self.conv1 = EdgeConv(num_features, hidden_dim)
        self.conv2 = EdgeConv(hidden_dim, hidden_dim)
        self.conv3 = EdgeConv(hidden_dim, hidden_dim)
        self.conv4 = EdgeConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, num_classes)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.lin1]:
            if hasattr(layer, "mlp"):
                for lin in layer.mlp:
                    if isinstance(lin, nn.Linear):
                        nn.init.xavier_uniform_(lin.weight, gain=1.0)
                        if lin.bias is not None:
                            nn.init.zeros_(lin.bias)
            elif hasattr(layer, "weight"):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index
        x = torch.cat([x, pos], dim=1)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        quotient_x, quotient_pos, quotient_edge_index = self.build_quotient_graph(x, pos, edge_index)

        quotient_x = F.relu(self.conv3(quotient_x, quotient_edge_index))
        quotient_x = F.relu(self.conv4(quotient_x, quotient_edge_index))

        graph_emb = global_add_pool(quotient_x, torch.zeros(quotient_x.size(0), dtype=torch.long, device=quotient_x.device))
        out = self.lin1(graph_emb).unsqueeze(0)

        return out.squeeze(0)

    def build_quotient_graph(self, x, pos, edge_index):
        kmeans = KMeans(n_clusters=self.num_regions, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(pos.cpu().detach().numpy())
        cluster_labels = torch.tensor(cluster_labels, device=x.device)

        quotient_x, quotient_pos = [], []
        for i in range(self.num_regions):
            mask = (cluster_labels == i)
            if mask.sum() == 0:
                region_feat = torch.zeros(x.size(1), device=x.device)
                region_coord = torch.zeros(pos.size(1), device=pos.device)
            else:
                region_feat = x[mask].mean(dim=0)
                region_coord = pos[mask].mean(dim=0)
            quotient_x.append(region_feat)
            quotient_pos.append(region_coord)

        quotient_x = torch.stack(quotient_x)
        quotient_pos = torch.stack(quotient_pos)

        knn = NearestNeighbors(n_neighbors=min(3, self.num_regions))
        knn.fit(quotient_pos.cpu().detach().numpy())
        _, indices = knn.kneighbors(quotient_pos.cpu().detach().numpy())

        connectivity = []
        for i in range(self.num_regions):
            for j in indices[i, 1:]:
                connectivity.append([i, j])
                connectivity.append([j, i])

        quotient_edge_index = torch.tensor(connectivity, dtype=torch.long, device=x.device).t().contiguous()

        return quotient_x, quotient_pos, quotient_edge_index
