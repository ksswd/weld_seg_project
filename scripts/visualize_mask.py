#!/usr/bin/env python3
"""Visualize masked points from preprocessed .npz files.

For each .npz in the input path (file or directory), this script loads features and
computes a mask using HighCurvatureMasker from train.mask_strategy. Masked points are
colored red, others gray, and the point cloud is saved as <basename>_mask.ply in the
same directory (or an optional output dir).

Usage:
    python scripts/visualize_mask.py data/processed/000.npz
    python scripts/visualize_mask.py data/processed --out data/vis
"""
import argparse
import os
import numpy as np
import open3d as o3d

from train.mask_strategy import HighCurvatureMasker

parser = argparse.ArgumentParser()
parser.add_argument('path', help='.npz file or directory with .npz files')
parser.add_argument('--out', help='output directory for ply files', default=None)
parser.add_argument('--mask_ratio', type=float, default=0.0005)
parser.add_argument('--random_frac', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

paths = []
if os.path.isdir(args.path):
    for f in sorted(os.listdir(args.path)):
        if f.endswith('.npz'):
            paths.append(os.path.join(args.path, f))
elif os.path.isfile(args.path) and args.path.endswith('.npz'):
    paths = [args.path]
else:
    raise SystemExit('Please provide a .npz file or directory')

out_dir = args.out if args.out is not None else os.path.dirname(paths[0])
os.makedirs(out_dir, exist_ok=True)

masker = HighCurvatureMasker(mask_ratio=args.mask_ratio, random_frac=args.random_frac, seed=args.seed)

for p in paths:
    d = np.load(p)
    if 'features' not in d:
        print('Skipping', p, 'no features key')
        continue
    feats = d['features']
    points = feats[..., :3]
    # reconstruct curvature if present
    if 'curvature' in d:
        curvature = d['curvature']
    else:
        # try to extract from features if available
        if feats.shape[1] >= 7:
            curvature = feats[:, 6:7]
        else:
            print('Skipping', p, 'no curvature available')
            continue

    # masker expects shape (B,N,1)
    curv_batch = curvature[None, ...].astype(np.float32)
    import torch
    mask = masker.generate_mask(torch.from_numpy(curv_batch))
    mask = mask[0, :, 0].numpy().astype(bool)

    colors = np.tile(np.array([0.5, 0.5, 0.5], dtype=np.float32), (points.shape[0], 1))
    colors[mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    out_path = os.path.join(out_dir, os.path.basename(p).replace('.npz', '_mask.ply'))
    o3d.io.write_point_cloud(out_path, pcd)
    print('Wrote', out_path)
