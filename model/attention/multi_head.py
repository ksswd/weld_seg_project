# weld_seg_project/model/attention/multi_head.py 多头几何注意力机制
import torch.nn as nn
import torch.nn.functional as F
from .anisotropic_dist import AnisotropicDistance
from .geom_bias import GeometryAttentionBias
from .side_gate_priori import SideGating as SideGatingPriori
from .side_gate_qwen import SideGating as SideGatingQwen

class MultiHeadGeometryAttention(nn.Module):
    def __init__(self, d_model, n_heads, config):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # ensure d_model is divisible by n_heads to avoid reshape issues later
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
                             " Adjust your model dimension or number of heads.")
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.aniso_dist = AnisotropicDistance(alpha0=config.ALPHA0, beta0=config.BETA0)
        self.geom_bias = GeometryAttentionBias(gamma=config.GAMMA, sigma=config.SIGMA)
        self.side_gate_type = config.SIDE_GATE_TYPE
        if self.side_gate_type == 'priori':
            self.side_gate = SideGatingPriori(weld_width_range=config.WELD_WIDTH_RANGE)
        elif self.side_gate_type == 'qwen':
            self.side_gate = SideGatingQwen(input_dim=d_model, output_dim=d_model)
        else:
            raise ValueError(f"Unsupported SIDE_GATE_TYPE: {config.SIDE_GATE_TYPE}. "
                             "Choose 'priori' or 'qwen'.")

    def forward(self, x, coordinate, principal_dir, curvature, density, normals, linearity):
        B, N, C = x.shape

        qkv = self.qkv_proj(x)
        # qkv.shape should be (B, N, 3 * d_model)
        qkv_last = qkv.size(-1)
        if qkv_last % 3 != 0:
            raise RuntimeError(f"qkv projection last dim ({qkv_last}) is not divisible by 3."
                               " Check d_model and qkv_proj configuration.")

        actual_d_model = qkv_last // 3
        if actual_d_model % self.n_heads != 0:
            raise RuntimeError(
                f"d_model ({actual_d_model}) is not divisible by n_heads ({self.n_heads})."
                " Ensure config.D_MODEL % config.N_HEADS == 0 and that input feature dim matches D_MODEL.")

        head_dim = actual_d_model // self.n_heads
        qkv = qkv.reshape(B, N, 3, self.n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attn_score shape: (B, n_heads, N, N)
        attn_score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        aniso_dist = self.aniso_dist(coordinate, principal_dir, linearity)
        geom_bias = self.geom_bias(curvature, density, normals, linearity, aniso_dist)
        
        # Apply bias and gate
        if self.side_gate_type == 'priori':
            side_gate = self.side_gate(coordinate, principal_dir, normals, density)
            attn_score = attn_score + geom_bias.unsqueeze(1)
            attn_score = attn_score * side_gate.unsqueeze(1)
            attn_weight = F.softmax(attn_score, dim=-1)
            attn_output = (attn_weight @ v).transpose(1, 2).reshape(B, N, C)
        elif self.side_gate_type == 'qwen':
            side_gate = self.side_gate(x)
            attn_score = attn_score + geom_bias.unsqueeze(1)
            attn_weight = F.softmax(attn_score, dim=-1)
            attn_output = (attn_weight @ v).transpose(1, 2).reshape(B, N, C)
            attn_output = attn_output * side_gate

        return self.out_proj(attn_output)