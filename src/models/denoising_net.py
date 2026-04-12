"""
Denoising U-Net backbone used by both HGD and DDM.

Architecture: MLP projector -> GAT downsampling stack -> bottleneck MLP
              -> GAT upsampling stack (with skip connections) -> MLP projector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GATConv
from .utils import create_activation, create_norm


class DenoisingUNet(nn.Module):
    """
    Graph U-Net with GAT layers.

    Down-path encodes node features at increasing abstraction; the up-path
    reconstructs with skip connections from corresponding down-path activations.
    Time embeddings are injected at every layer via residual addition.
    """

    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        out_dim: int,
        num_layers: int,
        nhead: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        norm: str,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        # Input projector: (in_dim + time) -> num_hidden
        self.mlp_in = MlpBlock(in_dim, num_hidden * 2, num_hidden,
                                norm=norm, activation=activation)
        self.mlp_middle = MlpBlock(num_hidden, num_hidden, num_hidden,
                                    norm=norm, activation=activation)
        self.mlp_out = MlpBlock(num_hidden, out_dim, out_dim,
                                 norm=norm, activation=activation)

        # Down-path GAT layers
        self.down_layers = nn.ModuleList()
        self.down_layers.append(
            GATConv(num_hidden, num_hidden // nhead, nhead,
                    feat_drop, attn_drop, negative_slope)
        )
        for _ in range(1, num_layers):
            self.down_layers.append(
                GATConv(num_hidden, num_hidden // nhead, nhead,
                        feat_drop, attn_drop, negative_slope)
            )

        # Up-path GAT layers (reversed order, first layer outputs full num_hidden)
        up = [GATConv(num_hidden, num_hidden, 1, feat_drop, attn_drop, negative_slope)]
        for _ in range(1, num_layers):
            up.append(
                GATConv(num_hidden, num_hidden // nhead, nhead,
                        feat_drop, attn_drop, negative_slope)
            )
        self.up_layers = nn.ModuleList(up[::-1])

    def forward(self, g, x_t: torch.Tensor, time_embed: torch.Tensor):
        """
        Parameters
        ----------
        g          : DGL graph
        x_t        : noisy node features, shape (N, in_dim)
        time_embed : time step embedding, shape (N, num_hidden)

        Returns
        -------
        out        : denoised prediction, shape (N, out_dim)
        hidden     : concatenated up-path activations for downstream use
        """
        h = self.mlp_in(x_t)

        # Down-path
        down_hidden = []
        for layer in self.down_layers:
            h = h + time_embed if h.ndim == 2 else h + time_embed.unsqueeze(1)
            h = layer(g, h).flatten(1)
            down_hidden.append(h)

        h = self.mlp_middle(h)

        # Up-path with skip connections
        out_hidden = []
        for i, layer in enumerate(self.up_layers):
            h = h + down_hidden[self.num_layers - i - 1]
            h = h + time_embed if h.ndim == 2 else h + time_embed.unsqueeze(1)
            h = layer(g, h).flatten(1)
            out_hidden.append(h)

        out = self.mlp_out(h)
        return out, torch.cat(out_hidden, dim=-1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class MlpBlock(nn.Module):
    """Two-layer MLP with a residual connection in the hidden space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = "layernorm", activation: str = "prelu"):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            create_norm(norm)(hidden_dim),
            create_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        ))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        return self.act(x)
