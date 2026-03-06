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
        raise RuntimeError(
            "plyfile is required to read scalar_seam from labeled PLYs. "
            "Install it in your conda env (e.g. `pip install plyfile`)."
        ) from e

    ply = PlyData.read(file_path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {file_path}")
    v = ply["vertex"].data
    names = v.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex is missing x/y/z: {file_path}(has {names})")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scalar = None
    # 兼容不同标注字段名（CloudCompare常见: scalar_label）
    for candidate in ("scalar_seam", "scalar_label", "seam", "label", "labels"):
        if candidate in names:
            scalar = np.asarray(v[candidate], dtype=np.float32)
            break
    return {"xyz": xyz, "scalar_seam": scalar}


def read_ply_file(file_path):
    """读取单个PLY文件，返回 numpy (N,3)"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"PLY file {file_path}does not contain (N,3) points.")
    return points


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
    """从csv文件加载特征（兼容新旧两种列名）"""
    df = pd.read_csv(file_path)

    # 新版预处理输出列（当前 preprocess.py）
    new_feat_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'curvature', 'density']
    # 旧版预处理输出列
    old_feat_cols = ['x', 'y', 'z', 'nx_norm', 'ny_norm', 'nz_norm', 'curvature_norm', 'density_norm']

    if set(new_feat_cols).issubset(df.columns):
        feat_cols = new_feat_cols
        normals_cols = ['nx', 'ny', 'nz']
        curvature_col = 'curvature'
        density_col = 'density'
    elif set(old_feat_cols).issubset(df.columns):
        feat_cols = old_feat_cols
        normals_cols = (
            ['nx_raw', 'ny_raw', 'nz_raw']
            if {'nx_raw', 'ny_raw', 'nz_raw'}.issubset(df.columns)
            else ['nx_norm', 'ny_norm', 'nz_norm']
        )
        curvature_col = 'curvature_raw' if 'curvature_raw' in df.columns else 'curvature_norm'
        density_col = 'local_density_raw' if 'local_density_raw' in df.columns else 'density_norm'
    else:
        raise KeyError(
            f"CSV列名不匹配。当前列: {list(df.columns)}。"
            f" 需要包含新版列 {new_feat_cols}或旧版列 {old_feat_cols}。"
        )

    feat = df[feat_cols].values.astype(np.float32)

    # coordinate
    coordinate = df[['x', 'y', 'z']].values.astype(np.float32)
    # normals
    normals = df[normals_cols].values.astype(np.float32)

    # curvature & density
    curvature = df[[curvature_col]].values.astype(np.float32)
    local_density = df[[density_col]].values.astype(np.float32)

    if {'principal_dir_x', 'principal_dir_y', 'principal_dir_z'}.issubset(df.columns):
        principal_dir = df[['principal_dir_x', 'principal_dir_y', 'principal_dir_z']].values.astype(np.float32)
    else:
        principal_dir = np.zeros_like(normals)

    if 'linearity' in df.columns:
        linearity = df[['linearity']].values.astype(np.float32)
    else:
        linearity = np.zeros((feat.shape[0], 1), dtype=np.float32)

    # supervised labels (optional)
    if 'label' in df.columns:
        labels = df[['label']].values.astype(np.float32)  # hard label
    else:
        labels = None

    if 'label_soft' in df.columns:
        labels_soft = df[['label_soft']].values.astype(np.float32)
    else:
        labels_soft = None

    return {
        'features': feat,                # (N, 8) -> 模型输入
        'coordinate': coordinate,
        'normals': normals,
        'curvature': curvature,
        'local_density': local_density,
        'principal_dir': principal_dir,
        'linearity': linearity,
        'labels': labels,                # (N,1) hard label or None
        'labels_soft': labels_soft       # (N,1) soft label or None
    }
