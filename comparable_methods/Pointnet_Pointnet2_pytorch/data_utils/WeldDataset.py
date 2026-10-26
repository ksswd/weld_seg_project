import os
import glob
import numpy as np
from torch.utils.data import Dataset


class WeldDataset(Dataset):
    """
    Weld semantic segmentation dataset for PointNet++.

    目录结构示例:
    data/weld/
    ├── train/
    │   ├── downsampled_labeled_3_aug_1_label_auto/
    │   │   ├── coord.npy
    │   │   ├── normal.npy
    │   │   ├── color.npy
    │   │   └── segment.npy
    │   └── ...
    ├── val/
    └── test/

    输出:
        points: (N, 9) -> [x, y, z, nx, ny, nz, x_norm, y_norm, z_norm]
        labels: (N,)
    """

    def __init__(
        self,
        root="data/weld",
        split="train",
        num_point=4096,
        use_normal=True,
        normalize_xyz=True,
        augment=False,
        ignore_label=None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.num_point = num_point
        self.use_normal = use_normal
        self.normalize_xyz = normalize_xyz
        self.augment = augment if split == "train" else False
        self.ignore_label = ignore_label

        self.split_dir = os.path.join(root, split)
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

        self.samples = sorted(
            [
                d for d in glob.glob(os.path.join(self.split_dir, "*"))
                if os.path.isdir(d)
            ]
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.split_dir}")

        self.num_classes = 2
        self.labelweights = self._compute_labelweights()

        print(
            f"[WeldDataset] split={self.split}, samples={len(self.samples)}, "
            f"num_point={self.num_point}, use_normal={self.use_normal}, "
            f"normalize_xyz={self.normalize_xyz}, augment={self.augment}"
        )

    def _compute_labelweights(self):
        hist = np.zeros(self.num_classes, dtype=np.float64)

        for sample_dir in self.samples:
            seg_path = os.path.join(sample_dir, "segment.npy")
            if not os.path.exists(seg_path):
                continue
            seg = np.load(seg_path).astype(np.int64)
            if self.ignore_label is not None:
                seg = seg[seg != self.ignore_label]
            if seg.size == 0:
                continue
            tmp, _ = np.histogram(seg, bins=np.arange(self.num_classes + 1))
            hist += tmp

        hist = hist + 1e-6
        labelweights = hist / np.sum(hist)
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        return labelweights.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def _normalize_xyz(self, xyz):
        """
        归一化到 roughly [-1, 1]
        """
        center = np.mean(xyz, axis=0)
        xyz_centered = xyz - center
        scale = np.max(np.linalg.norm(xyz_centered, axis=1))
        if scale < 1e-8:
            scale = 1.0
        xyz_norm = xyz_centered / scale
        return xyz_norm.astype(np.float32)

    def _sample_points(self, points, labels):
        n = points.shape[0]

        if n == self.num_point:
            choice = np.arange(n)
        elif n > self.num_point:
            choice = np.random.choice(n, self.num_point, replace=False)
        else:
            extra = np.random.choice(n, self.num_point - n, replace=True)
            choice = np.concatenate([np.arange(n), extra], axis=0)

        np.random.shuffle(choice)
        return points[choice], labels[choice]

    def _augment_points(self, xyz, normal=None):
        # random rotate around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cosval = np.cos(theta)
        sinval = np.sin(theta)
        rot = np.array(
            [
                [cosval, -sinval, 0.0],
                [sinval,  cosval, 0.0],
                [0.0,     0.0,    1.0],
            ],
            dtype=np.float32,
        )

        xyz = xyz @ rot.T
        if normal is not None:
            normal = normal @ rot.T

        # random scale
        scale = np.random.uniform(0.95, 1.05)
        xyz = xyz * scale

        # random shift
        shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
        xyz = xyz + shift

        # jitter
        jitter = np.clip(0.005 * np.random.randn(*xyz.shape), -0.02, 0.02).astype(np.float32)
        xyz = xyz + jitter

        return xyz, normal

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        coord_path = os.path.join(sample_dir, "coord.npy")
        normal_path = os.path.join(sample_dir, "normal.npy")
        segment_path = os.path.join(sample_dir, "segment.npy")

        if not os.path.exists(coord_path):
            raise FileNotFoundError(coord_path)
        if not os.path.exists(segment_path):
            raise FileNotFoundError(segment_path)

        xyz = np.load(coord_path).astype(np.float32)          # (M, 3)
        seg = np.load(segment_path).astype(np.int64)          # (M,)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"coord.npy shape should be (N,3), got {xyz.shape}")
        if seg.ndim != 1 or seg.shape[0] != xyz.shape[0]:
            raise ValueError(f"segment.npy shape mismatch: xyz={xyz.shape}, seg={seg.shape}")

        if self.use_normal:
            if not os.path.exists(normal_path):
                raise FileNotFoundError(normal_path)
            normal = np.load(normal_path).astype(np.float32)
            if normal.ndim != 2 or normal.shape[1] != 3 or normal.shape[0] != xyz.shape[0]:
                raise ValueError(f"normal.npy shape mismatch: {normal.shape}")
        else:
            normal = np.zeros_like(xyz, dtype=np.float32)

        if self.augment:
            xyz, normal = self._augment_points(xyz, normal)

        if self.normalize_xyz:
            xyz_norm = self._normalize_xyz(xyz)
        else:
            xyz_norm = xyz.copy()

        # PointNet++ uses the first three channels as coordinates for radius
        # queries.  Its radii (0.1/0.2/0.4/0.8) assume normalized coordinates,
        # not the raw millimetre-scale weld coordinates.  Keep nine channels
        # for checkpoint/model compatibility and avoid feeding unbounded raw
        # coordinates into the feature MLP.
        points = np.concatenate([xyz_norm, normal, xyz_norm], axis=1).astype(np.float32)

        points, seg = self._sample_points(points, seg)

        return points, seg
