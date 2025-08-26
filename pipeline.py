# pipeline.py
import pickle
from typing import List, Tuple

import numpy as np
import torch

from graph_utils import create_pyg_graph
from model import GLaRE
from train import graph_loader, train, test
import torch.nn as nn
import torch.optim as optim


def get_available_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU (CUDA)")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def load_pickle_data(pkl_path: str):
    """
    Expects keys: 'landmarks' (list of (N,3)),
                  'labels' (list of int),
                  'cnn_features' (list of (N,F))
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    landmarks_list = data["landmarks"]
    labels_list = data["labels"]
    features_list = data["cnn_features"]

    # Quick sanity
    assert len(landmarks_list) == len(labels_list) == len(features_list), \
        "landmarks, labels, cnn_features length mismatch"

    # Feature dim will be used for model config
    first_feat = np.asarray(features_list[0])
    assert first_feat.ndim == 2, f"cnn_features must be (N, F); got {first_feat.shape}"
    feat_dim = first_feat.shape[1]

    print(
        f"Loaded: {len(landmarks_list)} samples | "
        f"landmarks[0]: {np.asarray(landmarks_list[0]).shape} | "
        f"features[0]: {first_feat.shape} | label[0]: {labels_list[0]}"
    )
    return landmarks_list, labels_list, features_list, feat_dim


def build_graphs(
    landmarks_list: List[np.ndarray],
    labels_list: List[int],
    features_list: List[np.ndarray],
    k: int = 3,
) -> List["torch_geometric.data.Data"]:
    graphs = []
    for idx in range(len(landmarks_list)):
        try:
            g = create_pyg_graph(
                landmarks=landmarks_list[idx],
                label=labels_list[idx],
                features=features_list[idx],
                k=k,
                verbose=False,
            )
            # Important: ensure node features exist; model concatenates x with pos
            if g.x is None:
                raise ValueError(f"Sample {idx} has no node features (x is None).")
            graphs.append(g)
        except Exception as e:
            print(f"Skipping index {idx}: {e}")
            continue
    print(f"Built {len(graphs)} graphs (out of {len(landmarks_list)})")
    return graphs


def run_training_pipeline(
    graphs: List["torch_geometric.data.Data"],
    feat_dim: int,
    hidden_dim: int = 64,
    num_regions: int = 8,
    num_classes: int = 8,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
):
    """
    Trains GLaRE with batch_size=1 (the model assumes single-graph batches).
    """
    device = get_available_device()

    # Split into train/val/test DataLoaders
    train_loader, val_loader, test_loader = graph_loader(graphs)

    # num_features seen by the first EdgeConv is (node_feature_dim + 3 for pos)
    num_features = feat_dim + 3

    model = GLaRE(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_regions=num_regions,
    ).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train(model, device, train_loader, val_loader, epochs, criterion, optimizer)
    acc = test(model, device, test_loader)

    return model, acc
