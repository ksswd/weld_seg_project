import os
import re
import glob
import numpy as np
import pandas as pd
from plyfile import PlyData
from torch.utils.data import Dataset

FOLD_SPLITS = {
    "fold_1": {"test": [1, 5, 9, 14], "val": [2, 6, 10, 17], "train": [3, 4, 7, 8, 11, 12, 13, 15, 16, 18, 19]},
    "fold_2": {"test": [2, 6, 10, 17], "val": [3, 7, 11, 18], "train": [1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 19]},
    "fold_3": {"test": [3, 7, 11, 18], "val": [4, 8, 12, 19], "train": [1, 2, 5, 6, 9, 10, 13, 14, 15, 16, 17]},
    "fold_4": {"test": [4, 8, 12, 19], "val": [13, 15, 16], "train": [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 17, 18]},
    "fold_5": {"test": [13, 15, 16], "val": [1, 5, 9, 14], "train": [2, 3, 4, 6, 7, 8, 10, 11, 12, 17, 18, 19]},
    "fold_6": {"test": [3, 11], "val": [1, 2, 9, 10], "train": [1, 2, 9, 10]},
    "dataset": {"train": "train", "val": "validation", "test": "test"},
}

REQUIRED_COLUMNS = ["x", "y", "z", "nx", "ny", "nz", "label"]


def parse_group_id_from_name(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"^(\d+)\s*-", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse group id from filename: {filename}")


def read_ply_xyz_label(ply_path: str):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex'].data
    names = vertex.dtype.names
    if not {'x', 'y', 'z'}.issubset(names):
        raise ValueError(f'Missing x/y/z in {ply_path}')
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
    normal = np.zeros_like(xyz, dtype=np.float32)
    if {'nx', 'ny', 'nz'}.issubset(names):
        normal = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1).astype(np.float32)
    label = None
    for cand in ['scalar_label', 'label', 'pred']:
        if cand in names:
            label = vertex[cand].astype(np.int64)
            break
    if label is None:
        raise ValueError(f'No label column found in {ply_path}')
    label = np.clip(label, 0, 1).astype(np.int64)
    return xyz, normal, label


def sample_points(points, labels, num_point):
    n = points.shape[0]
    if n == num_point:
        idx = np.arange(n)
    elif n > num_point:
        idx = np.random.choice(n, num_point, replace=False)
    else:
        extra = np.random.choice(n, num_point - n, replace=True)
        idx = np.concatenate([np.arange(n), extra], axis=0)
    np.random.shuffle(idx)
    return points[idx], labels[idx]


def normalize_xyz(xyz):
    center = np.mean(xyz, axis=0)
    xyz_center = xyz - center
    scale = np.max(np.linalg.norm(xyz_center, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return (xyz_center / scale).astype(np.float32)


def augment_xyz_normal(xyz, normal):
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    xyz = xyz @ rot.T
    normal = normal @ rot.T
    scale = np.random.uniform(0.95, 1.05)
    xyz = xyz * scale
    shift = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
    xyz = xyz + shift
    jitter = np.clip(0.005 * np.random.randn(*xyz.shape), -0.02, 0.02).astype(np.float32)
    xyz = xyz + jitter
    return xyz, normal


class WeldCSVDataset(Dataset):
    def __init__(self, root, split="train", fold="dataset", num_point=8192, normalize=True, augment=False, use_soft_label=False, soft_threshold=0.5):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.fold = fold
        self.num_point = num_point
        self.normalize = normalize
        self.augment = augment if split == "train" else False
        self.use_soft_label = use_soft_label
        self.soft_threshold = soft_threshold

        if fold == "dataset":
            subdir = FOLD_SPLITS["dataset"][split]
            search_dir = os.path.join(root, subdir)
            self.files = sorted(glob.glob(os.path.join(search_dir, "**", "*.ply"), recursive=True))
        else:
            if fold not in FOLD_SPLITS:
                raise ValueError(f"Unknown fold: {fold}")
            search_dir = root
            all_ply = sorted(glob.glob(os.path.join(search_dir, "**", "*.ply"), recursive=True))
            group_set = set(FOLD_SPLITS[fold][split])
            self.files = [p for p in all_ply if parse_group_id_from_name(os.path.basename(p)) in group_set]

        if len(self.files) == 0:
            raise RuntimeError(f"No files selected for split={split}, fold={fold}, root={root}")

        self.num_classes = 2
        print(f"[WeldCSVDataset] split={split}, fold={fold}, files={len(self.files)}, num_point={num_point}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        xyz, normal, label = read_ply_xyz_label(path)

        if self.use_soft_label and "label_soft" in []:
            pass

        if self.augment:
            xyz, normal = augment_xyz_normal(xyz, normal)

        xyz_norm = normalize_xyz(xyz) if self.normalize else xyz.copy()
        points = np.concatenate([xyz, normal, xyz_norm], axis=1).astype(np.float32)
        points, label = sample_points(points, label, self.num_point)
        return points, label


class SingleWeldCSV(Dataset):
    def __init__(self, csv_path, num_point=8192, normalize=True, use_soft_label=False, soft_threshold=0.5):
        self.csv_path = csv_path
        self.num_point = num_point
        self.normalize = normalize
        self.use_soft_label = use_soft_label
        self.soft_threshold = soft_threshold

        if csv_path.lower().endswith('.ply'):
            self.xyz, self.normal, self.label = read_ply_xyz_label(csv_path)
        else:
            df = pd.read_csv(csv_path)
            missing = [c for c in ["x", "y", "z", "nx", "ny", "nz"] if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {csv_path}: {missing}")
            self.xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            self.normal = df[["nx", "ny", "nz"]].to_numpy(dtype=np.float32)
            if "label" in df.columns:
                label = df["label"].to_numpy(dtype=np.int64)
                self.label = np.clip(label, 0, 1).astype(np.int64)
            else:
                self.label = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        xyz = self.xyz.copy()
        normal = self.normal.copy()
        xyz_norm = normalize_xyz(xyz) if self.normalize else xyz.copy()
        points = np.concatenate([xyz, normal, xyz_norm], axis=1).astype(np.float32)

        n = points.shape[0]
        if n == self.num_point:
            sample_idx = np.arange(n)
        elif n > self.num_point:
            sample_idx = np.random.choice(n, self.num_point, replace=False)
        else:
            extra = np.random.choice(n, self.num_point - n, replace=True)
            sample_idx = np.concatenate([np.arange(n), extra], axis=0)

        pts = points[sample_idx]
        if self.label is None:
            lbl = np.full((self.num_point,), -1, dtype=np.int64)
        else:
            lbl = self.label[sample_idx]
        return pts, lbl, sample_idx, n
