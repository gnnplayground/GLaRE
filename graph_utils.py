import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def create_pyg_graph(landmarks, label=None, features=None, k=10, verbose=False):
    """
    Create a PyTorch Geometric Data object with k-NN connectivity from 3D landmarks.
    """
    landmarks = np.array(landmarks)

    if verbose:
        print("Input landmarks shape:", landmarks.shape)

    if landmarks.ndim == 3:
        if landmarks.shape[0] == 1:
            landmarks = landmarks[0]
        elif landmarks.shape[2] == 1:
            landmarks = landmarks[:, :, 0]
        else:
            raise ValueError(f"Unexpected shape {landmarks.shape}.")
    elif landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {landmarks.shape}")

    if verbose:
        print("Reshaped landmarks shape:", landmarks.shape)

    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(landmarks)
    _, indices = knn.kneighbors(landmarks)

    connectivity = [(i, j) for i in range(len(landmarks)) for j in indices[i, 1:]]
    edges = set((min(i, j), max(i, j)) for i, j in connectivity)

    edge_list = []
    for (i, j) in edges:
        edge_list.append([i, j])
        edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    pos = torch.tensor(landmarks, dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float) if features is not None else None
    y = torch.tensor([label], dtype=torch.long) if label is not None else None

    graph = Data(x=x, edge_index=edge_index, pos=pos, y=y)

    if verbose:
        print(f"Graph created with {graph.num_nodes} nodes and {graph.num_edges} edges")

    return graph
