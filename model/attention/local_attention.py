import torch
import torch.nn as nn
from .multi_head import MultiHeadGeometryAttention


class LocalAttention(nn.Module):
    """Thin wrapper that forces MultiHeadGeometryAttention into local (k-NN) mode.

    Use this when you want each query to attend only to its local k nearest neighbors.
    """

    def __init__(self, d_model: int, n_heads: int, config):
        super().__init__()
        self.attn = MultiHeadGeometryAttention(d_model=d_model, n_heads=n_heads, config=config)
        # force local mode
        self.attn.attention_mode = 'local'

    def forward(self, x, principal_dir, curvature, density, normals, linearity):
        return self.attn(x, principal_dir, curvature, density, normals, linearity)
