"""
model.py
────────
Defines the GCN model architecture.
Kept separate so the architecture can be swapped without touching training code.

Architecture:
    Input  (num_features)
      ↓
    [GCNConv → ReLU → Dropout]  × (num_layers - 1)
      ↓
    GCNConv → log_softmax
      ↓
    Output (num_classes)

Usage (imported):
    from model import GCN, build_model
    model = build_model(num_features=14, num_classes=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import config


class GCN(torch.nn.Module):
    """
    Multi-layer Graph Convolutional Network.

    Architecture:
        Input  (num_features)
          ↓
        [GCNConv → ReLU → Dropout]  × (num_layers - 1)
          ↓
        GCNConv → log_softmax
          ↓
        Output (num_classes)
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        num_layers:      int   = 2,
        dropout:         float = 0.5,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.num_layers = num_layers
        self.dropout    = dropout

        # Build conv layers
        self.convs = nn.ModuleList()

        # First layer: in_channels → hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Middle layers (if num_layers > 2): hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final layer: hidden_channels → out_channels (no ReLU)
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        # Hidden layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


def build_model(num_features: int, num_classes: int) -> GCN:
    """
    Instantiate GCN using hyperparameters from config.

    Parameters
    ----------
    num_features : int  — number of node input features
    num_classes  : int  — number of output classes

    Returns
    -------
    GCN instance (not yet moved to device)
    """
    model = GCN(
        in_channels     = num_features,
        hidden_channels = config.HIDDEN_DIM,
        out_channels    = num_classes,
        num_layers      = config.NUM_LAYERS,
        dropout         = config.DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] GCN({num_features} → {config.HIDDEN_DIM} × {config.NUM_LAYERS - 1} → {num_classes})")
    print(f"[model] Trainable parameters : {total_params:,}")
    print(f"[model] Layers               : {config.NUM_LAYERS}")

    return model
