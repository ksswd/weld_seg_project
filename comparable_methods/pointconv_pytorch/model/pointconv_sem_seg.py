import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pointconv_util import PointConvDensitySetAbstraction, square_distance, index_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        bsz, n, _ = xyz1.shape
        _, s, _ = xyz2.shape

        if s == 1:
            interpolated_points = points2.repeat(1, n, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(bsz, n, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointConvSemSeg(nn.Module):
    def __init__(self, num_classes=2, input_dim=9):
        super().__init__()
        feat_dim = input_dim - 3

        self.sa1 = PointConvDensitySetAbstraction(1024, 32, feat_dim + 3, [32, 32, 64], bandwidth=0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(256, 32, 64 + 3, [64, 64, 128], bandwidth=0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(64, 32, 128 + 3, [128, 128, 256], bandwidth=0.4, group_all=False)
        self.sa4 = PointConvDensitySetAbstraction(16, 32, 256 + 3, [256, 256, 512], bandwidth=0.8, group_all=False)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        # x: [B, 9, N]
        l0_xyz = x[:, :3, :]
        l0_points = x[:, 3:, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


def get_model(num_classes=2, input_dim=9):
    return PointConvSemSeg(num_classes=num_classes, input_dim=input_dim)


def get_loss(pred, target):
    return F.nll_loss(pred, target)
