# weld_seg_project/model/attention/local_attention.py
"""
经典点云局部注意力机制 - 基于k-NN
每个点只关注其空间上最近的k个邻居点
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttention(nn.Module):
    """
    基于k-NN的局部自注意力

    特点：
    - 每个query点只attend到其k个最近邻
    - 减少计算复杂度：O(N*k) vs O(N^2)
    - 更好地捕获局部几何结构
    """

    def __init__(self, d_model: int, n_heads: int, config):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k_neighbors = getattr(config, 'LOCAL_K_NEIGHBORS', 16)  # 默认16个邻居

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, coordinate=None, principal_dir=None, curvature=None, density=None, normals=None, linearity=None):
        """
        Args:
            x: (B, N, C) 输入特征
            其他参数保留接口兼容性，但本模块不使用

        Returns:
            (B, N, C) 注意力输出
        """
        B, N, C = x.shape
        k = min(self.k_neighbors, N)  # 确保k不超过点数

        # 提取xyz坐标用于计算距离
        xyz = coordinate  # (B, N, 3)

        # 计算k-NN索引
        knn_idx = self._knn(xyz, k)  # (B, N, k)

        # 投影Q, K, V
        q = self.q_proj(x)  # (B, N, d_model)
        k_feat = self.k_proj(x)  # (B, N, d_model)
        v = self.v_proj(x)  # (B, N, d_model)

        # 根据k-NN索引gather邻居的K和V
        # knn_idx: (B, N, k) -> 扩展到 (B, N, k, d_model)
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)

        # (B, N, k, d_model)
        k_neighbors = torch.gather(
            k_feat.unsqueeze(2).expand(-1, -1, k, -1),
            dim=1,
            index=knn_idx_expanded.transpose(1, 2).reshape(B, -1, self.d_model).unsqueeze(2).expand(-1, -1, k, -1)
        )
        # 更简单的gather方式
        k_neighbors = torch.gather(
            k_feat.unsqueeze(1).expand(-1, N, -1, -1),  # (B, N, N, d_model)
            dim=2,
            index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)  # (B, N, k, d_model)
        )
        v_neighbors = torch.gather(
            v.unsqueeze(1).expand(-1, N, -1, -1),  # (B, N, N, d_model)
            dim=2,
            index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)  # (B, N, k, d_model)
        )

        # Reshape for multi-head attention
        # q: (B, N, n_heads, head_dim) -> (B, n_heads, N, head_dim)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        # k_neighbors, v_neighbors: (B, N, k, d_model) -> (B, n_heads, N, k, head_dim)
        k_neighbors = k_neighbors.view(B, N, k, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v_neighbors = v_neighbors.view(B, N, k, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # 计算注意力分数: (B, n_heads, N, 1, head_dim) @ (B, n_heads, N, head_dim, k)
        # -> (B, n_heads, N, 1, k) -> (B, n_heads, N, k)
        attn_score = torch.matmul(
            q.unsqueeze(3),  # (B, n_heads, N, 1, head_dim)
            k_neighbors.transpose(-2, -1)  # (B, n_heads, N, head_dim, k)
        ).squeeze(3) * self.scale  # (B, n_heads, N, k)

        attn_weight = F.softmax(attn_score, dim=-1)  # (B, n_heads, N, k)

        # 加权求和: (B, n_heads, N, k) @ (B, n_heads, N, k, head_dim) -> (B, n_heads, N, head_dim)
        attn_output = torch.matmul(
            attn_weight.unsqueeze(3),  # (B, n_heads, N, 1, k)
            v_neighbors  # (B, n_heads, N, k, head_dim)
        ).squeeze(3)  # (B, n_heads, N, head_dim)

        # Reshape back: (B, n_heads, N, head_dim) -> (B, N, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.d_model)

        return self.out_proj(attn_output)

    def _knn(self, xyz, k):
        """
        计算k近邻索引

        Args:
            xyz: (B, N, 3) 点坐标
            k: 邻居数量

        Returns:
            (B, N, k) 每个点的k个最近邻索引
        """
        # 计算距离矩阵 (B, N, N)
        dist = torch.cdist(xyz, xyz)  # 欧氏距离

        # 取最近的k个（包括自己）
        _, idx = torch.topk(dist, k, dim=-1, largest=False)  # (B, N, k)

        return idx
