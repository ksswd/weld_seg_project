# weld_seg_project/model/attention/global_attention.py
"""
经典点云全局注意力机制
所有点之间相互attend，捕获全局上下文信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    """
    标准全局自注意力

    特点：
    - 每个点attend到所有其他点
    - 计算复杂度：O(N^2)
    - 捕获长距离依赖和全局上下文
    """

    def __init__(self, d_model: int, n_heads: int, config=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, principal_dir=None, curvature=None, density=None, normals=None, linearity=None):
        """
        Args:
            x: (B, N, C) 输入特征
            其他参数保留接口兼容性，但本模块不使用

        Returns:
            (B, N, C) 注意力输出
        """
        B, N, C = x.shape

        # 投影Q, K, V
        q = self.q_proj(x)  # (B, N, d_model)
        k = self.k_proj(x)  # (B, N, d_model)
        v = self.v_proj(x)  # (B, N, d_model)

        # Reshape for multi-head attention
        # (B, N, d_model) -> (B, N, n_heads, head_dim) -> (B, n_heads, N, head_dim)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数: (B, n_heads, N, head_dim) @ (B, n_heads, head_dim, N)
        # -> (B, n_heads, N, N)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Softmax
        attn_weight = F.softmax(attn_score, dim=-1)  # (B, n_heads, N, N)

        # 加权求和: (B, n_heads, N, N) @ (B, n_heads, N, head_dim) -> (B, n_heads, N, head_dim)
        attn_output = torch.matmul(attn_weight, v)

        # Reshape back: (B, n_heads, N, head_dim) -> (B, N, n_heads, head_dim) -> (B, N, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.d_model)

        return self.out_proj(attn_output)
