import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import json
import re
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


# =========== ModelNet40 =================
class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition  # Here the new given partition will cover the 'train'

    def __getitem__(self, item):  # indice of the pts or label
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# =========== ShapeNet Part =================
class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False):
        self.npoints = npoints
        self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal

    def __len__(self):
        return len(self.datapath)


class WeldPartDataset(Dataset):
    def __init__(self, root, split="train", npoints=2048, normalize=True, use_soft_label=False, soft_threshold=0.5, return_meta=False):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = root
        self.npoints = npoints
        self.normalize = normalize
        self.use_soft_label = use_soft_label
        self.soft_threshold = soft_threshold
        self.return_meta = return_meta

        split_dir_name = 'validation' if split == 'val' else split
        split_dir = os.path.join(root, split_dir_name)
        all_csv = sorted(glob.glob(os.path.join(split_dir, "**", "*.csv"), recursive=True))
        if len(all_csv) == 0:
            raise RuntimeError(f"No csv files found in: {split_dir}")

        self.files = all_csv
        print(f"[WeldPartDataset] split={split}, files={len(self.files)}, npoints={npoints}")

    @staticmethod
    def _parse_group_id(filename: str) -> int:
        stem = os.path.splitext(filename)[0]
        m = re.match(r"^downsampled_labeled_(\d+)", stem)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)", stem)
        if m:
            return int(m.group(1))
        raise ValueError(f"Cannot parse group id from filename: {filename}")

    @staticmethod
    def _normalize_xyz(xyz):
        center = np.mean(xyz, axis=0)
        xyz = xyz - center
        scale = np.max(np.linalg.norm(xyz, axis=1))
        if scale < 1e-8:
            scale = 1.0
        return (xyz / scale).astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        df = pd.read_csv(path)
        required_cols = ["x", "y", "z", "nx", "ny", "nz", "label"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        normal = df[["nx", "ny", "nz"]].to_numpy(dtype=np.float32)

        if self.use_soft_label and "label_soft" in df.columns:
            seg = (df["label_soft"].to_numpy(dtype=np.float32) >= self.soft_threshold).astype(np.int32)
        else:
            seg = df["label"].to_numpy(dtype=np.int32)
        seg = np.clip(seg, 0, 1).astype(np.int32)

        if self.normalize:
            xyz = self._normalize_xyz(xyz)

        point_set = xyz
        cls = np.array([0]).astype(np.int32)
        if self.return_meta:
            file_path = path
            sampled_indices = np.arange(len(seg), dtype=np.int64)
            return point_set, cls, seg, normal, file_path, sampled_indices
        return point_set, cls, seg, normal


if __name__ == '__main__':
    train = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    test = PartNormalDataset(npoints=2048, split='test', normalize=False)
    for data, label, _, _ in train:
        print(data.shape)
        print(label.shape)
