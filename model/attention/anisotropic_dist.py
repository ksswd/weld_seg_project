# weld_seg_project/model/attention/anisotropic_dist.py 各向异性距离计算
import torch
import torch.nn as nn

class AnisotropicDistance(nn.Module):
    def __init__(self, alpha0=2.0, beta0=0.5):
        super().__init__()
        self.alpha0 = alpha0
        self.beta0 = beta0

    def forward(self, points, principal_dir, linearity):
        B, N, _ = points.shape

        # Compute pairwise quantities in row-wise chunks to avoid allocating large (B, N, N, 3)
        device = points.device
        dtype = points.dtype
        out = torch.empty((B, N, N), device=device, dtype=dtype)

        # chunk size -- tuneable: smaller -> lower peak memory, but more loops
        CHUNK = 256

        t = principal_dir.unsqueeze(2)  # (B, N, 1, 3)

        for i in range(0, N, CHUNK):
            i2 = min(N, i + CHUNK)

            # compute differences for rows i:i2: shape (B, i2-i, N, 3)
            point_i = points[:, i:i2, :].unsqueeze(2)  # (B, i2-i, 1, 3)
            point_all = points.unsqueeze(1)            # (B, 1, N, 3)
            point_diff = point_i - point_all            # (B, i2-i, N, 3)

            # project along principal direction
            t_i = t[:, i:i2, :, :]                      # (B, i2-i, 1, 3)
            along_t = torch.matmul(point_diff, t_i.transpose(2, 3)).squeeze(-1)  # (B, i2-i, N)

            # normal plane distance
            normal_plane = point_diff - along_t.unsqueeze(-1) * t_i  # (B, i2-i, N, 3)
            normal_dist = torch.norm(normal_plane, dim=-1)  # (B, i2-i, N)

            # alpha, beta for these rows: (B, i2-i, 1)
            alpha_i = self.alpha0 * (1 + linearity[:, i:i2, :])
            beta_i = self.beta0 * (1 - linearity[:, i:i2, :])

            # distance block
            dist_block = alpha_i * (normal_dist ** 2) + beta_i * (along_t ** 2)  # (B, i2-i, N)

            out[:, i:i2, :] = dist_block

        return out