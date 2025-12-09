# weld_seg_project/model/attention/side_gate.py 侧门控机制
import torch
import torch.nn as nn

class SideGating(nn.Module):
    def __init__(self, weld_width_range=[0.005, 0.02]):
        super().__init__()
        self.w_min, self.w_max = weld_width_range
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.a3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, points, principal_dir, normals, density):
        point_diff = points.unsqueeze(2) - points.unsqueeze(1)

        t = principal_dir.unsqueeze(2)
        along_dist = torch.matmul(point_diff, t.transpose(2, 3)).squeeze(-1)
        normal_dist = torch.norm(point_diff - along_dist.unsqueeze(-1) * t, dim=-1)

        # Density gap (current point is in a valley compared to neighbor)
        density_gap = density.unsqueeze(2) - density.unsqueeze(1)

        # Normal distance within weld width
        clipped_normal_dist = torch.clamp(normal_dist, self.w_min, self.w_max)

        # Normal consistency
        normal_dot = torch.matmul(normals, normals.transpose(1, 2))

        gate = self.a1 * density_gap.squeeze(-1) + self.a2 * clipped_normal_dist - self.a3 * (1 - normal_dot)
        gate = torch.sigmoid(gate)
        return gate