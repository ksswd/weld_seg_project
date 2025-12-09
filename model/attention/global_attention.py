import torch.nn as nn
from .multi_head import MultiHeadGeometryAttention


class GlobalAttention(nn.Module):
    """Thin wrapper that forces MultiHeadGeometryAttention into global mode.

    This keeps the original geometry-aware attention implementation but
    ensures the attention_mode is 'global' for every instance.
    """

    def __init__(self, d_model: int, n_heads: int, config):
        super().__init__()
        # instantiate underlying attention
        self.attn = MultiHeadGeometryAttention(d_model=d_model, n_heads=n_heads, config=config)
        # force global mode regardless of provided config
        self.attn.attention_mode = 'global'

    def forward(self, x, principal_dir, curvature, density, normals, linearity):
        return self.attn(x, principal_dir, curvature, density, normals, linearity)
