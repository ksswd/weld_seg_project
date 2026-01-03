# weld_seg_project/utils/io_utils.py 输入输出相关
import os
import numpy as np
import open3d as o3d
from glob import glob
import pandas as pd
from typing import Optional, Dict, Any


def read_ply_with_scalar_seam(file_path: str) -> Dict[str, Any]:
    """
    Read a PLY while preserving custom scalar fields (e.g. scalar_seam).

    NOTE: Open3D's read_point_cloud typically drops custom per-vertex properties,
    which is why labeled PLYs can lose the seam label during preprocessing.

    Returns:
        {
          'xyz': (N,3) float32,
          'scalar_seam': (N,) float32 or None
        }
    """
    try:
        from plyfile import PlyData  # type: ignore
    except Exception as e:
        raise RuntimeError("plyfile is required to read scalar_seam from labeled PLYs. "
                           "Install it in your conda env (e.g. `pip install plyfile`).") from e

    ply = PlyData.read(file_path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {file_path}")
    v = ply["vertex"].data
    names = v.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex is missing x/y/z: {file_path} (has {names})")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scalar = None
    if "scalar_seam" in names:
        scalar = np.asarray(v["scalar_seam"], dtype=np.float32)
    return {"xyz": xyz, "scalar_seam": scalar}

def read_ply_file(file_path):
    """读取单个PLY文件"""
    # Prefer plyfile for labeled PLYs so we don't lose scalar_seam.
    # Fallback to Open3D for unlabeled or if scalar field isn't present.
    try:
        data = read_ply_with_scalar_seam(file_path)
        if data.get("scalar_seam") is not None:
            return data
    except Exception:
        pass
    ply = o3d.io.read_point_cloud(file_path)
    return ply

def read_all_ply_from_dir(dir_path):
    """读取目录中所有PLY文件"""
    ply_files = glob(os.path.join(dir_path, "*.ply"))
    point_clouds = []
    for file in ply_files:
        pc = read_ply_file(file)
        point_clouds.append(pc)
    return point_clouds, ply_files

def save_features_to_npz(features, file_path):
    """将预处理后的特征保存为NPZ文件"""
    np.savez_compressed(file_path, features=features)

def load_features_from_npz(file_path):
    """从NPZ文件加载特征"""
    data = np.load(file_path)
    return data['features']

def load_features_from_csv(file_path):
    """从csv文件加载特征"""
    df = pd.read_csv(file_path)
    feat = df[[
                'x', 'y', 'z',
                'nx_norm', 'ny_norm', 'nz_norm',
                'curvature_norm', 'density_norm'
            ]].values.astype(np.float32)
    # coordinate
    coordinate = df[['x', 'y', 'z']].values.astype(np.float32)
    # normals
    normals = df[['nx_raw', 'ny_raw', 'nz_raw']].values.astype(np.float32)

    # curvature & density
    curvature = df[['curvature_raw']].values.astype(np.float32)
    local_density = df[['local_density_raw']].values.astype(np.float32)

    if {'principal_dir_x', 'principal_dir_y', 'principal_dir_z'}.issubset(df.columns):
        principal_dir = df[
            ['principal_dir_x', 'principal_dir_y', 'principal_dir_z']
        ].values.astype(np.float32)
    else:
        principal_dir = np.zeros_like(normals)

    if 'linearity' in df.columns:
        linearity = df[['linearity']].values.astype(np.float32)
    else:
        linearity = np.zeros((feat.shape[0], 1), dtype=np.float32)
    # supervised label (optional)
    if 'label' in df.columns:
        labels = df[['label']].values.astype(np.float32)
    else:
        labels = None
    return {
        'features': feat,                  # (N, 8)  -> 模型输入
        'coordinate': coordinate,           # coordinate
        'normals': normals,                # raw normals
        'curvature': curvature,            # raw curvature
        'local_density': local_density,    # raw density
        'principal_dir': principal_dir,    # raw PCA direction
        'linearity': linearity,             # raw linearity
        'labels': labels                    # (N,1) or None
    }
