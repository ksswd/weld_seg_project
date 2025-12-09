#!/usr/bin/env python3
"""Convert one or more .npz files to .ply for visualization.

Usage:
    python scripts/npz_to_ply.py /path/to/file_or_dir [--key preds]

If a directory is given, all .npz files inside will be converted.
"""
import argparse
import os
import numpy as np
import open3d as o3d
from glob import glob

def save_npz_to_ply(npz_path, ply_path=None, color_by=None):
    """Convert a .npz (preprocessed features or features+preds) into a .ply file for viewing.

    Behavior:
    - If the npz contains 'features' key with shape (N, C) and C >= 3, the first 3 columns are taken as XYZ.
    - If the npz has an explicit 'predictions' or 'preds' key, or the last column of 'features' looks binary,
      those values are used to color weld points red and others gray by default.
    - If 'normals' exist in the archive, they're attached to the PLY.

    Args:
        npz_path (str): Path to the .npz file.
        ply_path (str or None): Path to write the .ply file. If None, uses same basename with .ply.
        color_by (str or None): Key to use for coloring. If None, auto-detect ('predictions','preds', or last feature col).
    """
    data = np.load(npz_path)

    # Determine points
    if 'features' in data:
        feat = data['features']
        if feat.ndim != 2 or feat.shape[1] < 3:
            raise ValueError('features must be an (N, C>=3) array')
        points = feat[..., :3].astype(np.float64)
    elif 'points' in data:
        points = data['points'].astype(np.float64)
    else:
        raise KeyError('No "features" or "points" key found in npz file')

    N = points.shape[0]

    # Determine coloring
    preds = None
    if color_by is not None and color_by in data:
        preds = data[color_by]
    else:
        # look for common prediction key names
        for k in ('predictions', 'preds', 'labels'):
            if k in data:
                preds = data[k]
                break

    if preds is None and 'features' in data and data['features'].shape[1] > 3:
        # heuristically check if last column looks binary (0/1)
        last_col = data['features'][..., -1]
        if np.logical_or(last_col == 0, last_col == 1).all():
            preds = last_col

    # Prepare colors: default grey, weld predictions red
    if preds is None:
        colors = np.tile(np.array([0.5, 0.5, 0.5], dtype=np.float64), (N, 1))
    else:
        preds = np.asarray(preds).reshape(-1)
        if preds.shape[0] != N:
            # try flattening if preds had extra dims (e.g., (N,1))
            try:
                preds = preds.reshape(N)
            except Exception:
                raise ValueError('Prediction array length does not match number of points')
        colors = np.zeros((N, 3), dtype=np.float64)
        # default non-weld grey
        colors[:] = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        # weld -> red
        mask = preds.astype(bool)
        colors[mask] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # attach normals if available
    if 'normals' in data:
        try:
            norms = data['normals']
            if norms.shape[0] == N and norms.shape[1] >= 3:
                pcd.normals = o3d.utility.Vector3dVector(norms[..., :3].astype(np.float64))
        except Exception:
            pass

    if ply_path is None:
        ply_path = os.path.splitext(npz_path)[0] + '.ply'

    o3d.io.write_point_cloud(ply_path, pcd)
    return ply_path


parser = argparse.ArgumentParser()
parser.add_argument('path', help='.npz file or directory containing .npz files')
parser.add_argument('--key', help='which key to use for predictions (optional)', default=None)
args = parser.parse_args()

paths = []
if os.path.isdir(args.path):
    for f in sorted(os.listdir(args.path)):
        if f.endswith('_pred.npz'):
            paths.append(os.path.join(args.path, f))
elif os.path.isfile(args.path) and args.path.endswith('.npz'):
    paths = [args.path]
else:
    raise SystemExit('Please provide a .npz file or directory containing .npz files')

for p in paths:
    out = save_npz_to_ply(p, color_by=args.key)
    print(f'Wrote {out}')

