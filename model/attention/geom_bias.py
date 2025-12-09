# weld_seg_project/model/attention/geom_bias.py 几何注意力偏置
import torch
import torch.nn as nn

class GeometryAttentionBias(nn.Module):
    def __init__(self, gamma=1.0, sigma=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        # learnable curvature scaling to increase priority of curvature feature
        # initialize to 1.0 (no change) but training can increase/decrease
        self.curvature_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, curvature, density, normals, linearity, anisotropic_dist):
        B, N, _ = curvature.shape

        # To avoid allocating large (B, N, N) temporaries for each term, compute the bias
        # in row-wise chunks: for rows i:i2 compute (B, i2-i, N) blocks and write into output.
        device = curvature.device
        dtype = curvature.dtype

        # chunk size controls memory vs compute tradeoff; tune if still OOM
        CHUNK = 256
        out = torch.empty((B, N, N), device=device, dtype=dtype)

        for i in range(0, N, CHUNK):
            i2 = min(N, i + CHUNK)

            # curvature similarity block: shape (B, i2-i, N)
            # optionally scale curvature feature to increase its influence
            # curvature shape: (B, N, 1)
            cur = curvature * self.curvature_scale
            cur_i = cur[:, i:i2, :]
            cur_diff = cur_i.unsqueeze(2) - cur.unsqueeze(1)
            cur_sim = torch.exp(-(cur_diff ** 2) / (2 * self.gamma ** 2)).squeeze(-1)

            # normals block: (B, i2-i, 3) @ (B, 3, N) -> (B, i2-i, N)
            norm_i = normals[:, i:i2, :]
            norm_dot = torch.matmul(norm_i, normals.transpose(1, 2))
            norm_cons = (norm_dot + 1) / 2

            # density block: (B, i2-i, 1) + (B, 1, N) -> (B, i2-i, N)
            den_i = density[:, i:i2, :]
            den_bias = -self.sigma * (den_i.unsqueeze(2) + density.unsqueeze(1)).squeeze(-1)

            # linearity block: (B, i2-i, 1) * (B, 1, N) -> (B, i2-i, N)
            lin_i = linearity[:, i:i2, :]
            lin_bias = (lin_i.unsqueeze(2) * linearity.unsqueeze(1)).squeeze(-1)
            # lin_bias = 0

            # distance block: anisotropic_dist is expected to be (B, N, N)
            dist_block = -anisotropic_dist[:, i:i2, :]

            out[:, i:i2, :] = cur_sim + norm_cons + den_bias + lin_bias + dist_block

        return out