import os
import glob
import numpy as np
from torch.utils.data import Dataset


class WeldDataset(Dataset):
    """Weld semantic segmentation dataset for PointConv/PointNet++ style input.

    Expected structure:
      root/
        train/*.ply
        validation/*.ply
        test/*.ply

    Each .ply is ASCII and contains fields:
      x, y, z, scalar_label

    We build 9-D input for compatibility with the existing PointConv model:
      [x, y, z, nx, ny, nz, x_norm, y_norm, z_norm]
    where normals are zero-filled because this dataset does not provide normals.
    """

    def __init__(
        self,
        root="data/dataset",
        split="train",
        num_point=4096,
        use_normal=True,
        normalize_xyz=True,
        augment=False,
        ignore_label=None,
        return_name=False,
    ):
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.root = root
        self.split = split
        self.num_point = num_point
        self.use_normal = use_normal
        self.normalize_xyz = normalize_xyz
        self.augment = augment if split == "train" else False
        self.ignore_label = ignore_label
        self.return_name = return_name

        self.split_dir = os.path.join(root, split)
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

        self.samples = sorted(glob.glob(os.path.join(self.split_dir, "*.ply")))
        if len(self.samples) == 0:
            raise RuntimeError(f"No .ply files found in {self.split_dir}")

        self.num_classes = 2
        self.labelweights = self._compute_labelweights()

        print(
            f"[WeldDataset] split={self.split}, samples={len(self.samples)}, "
            f"num_point={self.num_point}, use_normal={self.use_normal}, "
            f"normalize_xyz={self.normalize_xyz}, augment={self.augment}"
        )

    def __len__(self):
        return len(self.samples)

    def _compute_labelweights(self):
        hist = np.zeros(self.num_classes, dtype=np.float64)
        for ply_path in self.samples:
            _, seg = self._read_ply(ply_path)
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

    @staticmethod
    def _read_ply(ply_path):
        with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
            header = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Invalid ply file (no end_header): {ply_path}")
                header.append(line.strip())
                if line.strip() == "end_header":
                    break

            vertex_count = None
            properties = []
            for line in header:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    if len(parts) >= 3:
                        properties.append(parts[-1])

            if vertex_count is None:
                raise ValueError(f"No vertex count found in ply header: {ply_path}")

            prop_to_idx = {name: i for i, name in enumerate(properties)}
            required = {"x", "y", "z", "scalar_label"}
            if not required.issubset(prop_to_idx.keys()):
                raise ValueError(f"PLY must contain x/y/z/scalar_label fields, got {properties} in {ply_path}")

            points = np.zeros((vertex_count, 3), dtype=np.float32)
            labels = np.zeros((vertex_count,), dtype=np.int64)
            for i in range(vertex_count):
                parts = f.readline().strip().split()
                if len(parts) < len(properties):
                    raise ValueError(f"Invalid vertex line in {ply_path} at row {i}")
                points[i, 0] = float(parts[prop_to_idx["x"]])
                points[i, 1] = float(parts[prop_to_idx["y"]])
                points[i, 2] = float(parts[prop_to_idx["z"]])
                labels[i] = int(float(parts[prop_to_idx["scalar_label"]]))

        return points, labels

    @staticmethod
    def _normalize_xyz(xyz):
        center = np.mean(xyz, axis=0)
        xyz_centered = xyz - center
        scale = np.max(np.linalg.norm(xyz_centered, axis=1))
        if scale < 1e-8:
            scale = 1.0
        return (xyz_centered / scale).astype(np.float32)

    def _sample_points(self, points, labels):
        n = points.shape[0]
        choice = np.arange(n)
        return points[choice], labels[choice]

    @staticmethod
    def _augment_points(xyz, normal=None):
        theta = np.random.uniform(0, 2 * np.pi)
        cosval = np.cos(theta)
        sinval = np.sin(theta)
        rot = np.array(
            [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        xyz = xyz @ rot.T
        if normal is not None:
            normal = normal @ rot.T

        scale = np.random.uniform(0.95, 1.05)
        xyz = xyz * scale

        shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
        xyz = xyz + shift

        jitter = np.clip(0.005 * np.random.randn(*xyz.shape), -0.02, 0.02).astype(np.float32)
        xyz = xyz + jitter

        return xyz, normal

    def __getitem__(self, idx):
        ply_path = self.samples[idx]
        sample_name = os.path.splitext(os.path.basename(ply_path))[0]

        xyz, seg = self._read_ply(ply_path)

        if self.ignore_label is not None:
            valid = seg != self.ignore_label
            xyz = xyz[valid]
            seg = seg[valid]

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"PLY xyz shape should be (N,3), got {xyz.shape}")
        if seg.ndim != 1 or seg.shape[0] != xyz.shape[0]:
            raise ValueError(f"label shape mismatch: xyz={xyz.shape}, seg={seg.shape}")

        normal = np.zeros_like(xyz, dtype=np.float32)

        if self.augment:
            xyz, normal = self._augment_points(xyz, normal)

        if self.normalize_xyz:
            xyz_norm = self._normalize_xyz(xyz)
        else:
            xyz_norm = xyz.copy()

        points = np.concatenate([xyz, normal, xyz_norm], axis=1).astype(np.float32)
        points, seg = self._sample_points(points, seg)

        if self.return_name:
            return points, seg, sample_name
        return points, seg
