#!/usr/bin/env python3
"""
Inject per-point labels into processed .npz files.

Usage:
  python scripts/add_labels_to_npz.py [--overwrite]

Behavior:
 - Reads label files from Config.LABEL_DATA_DIR. Supported formats: .npy, .npz, .csv, .txt, .ply
 - For each label file, finds matching processed .npz by basename in Config.PROCESSED_DATA_DIR
 - Validates the label length matches the number of points in the .npz ('features')
 - Backs up the original .npz to .orig if not already backed up, then writes a new .npz containing existing arrays plus key 'labels'

This is a small utility to prepare supervised labels for training.
"""
import os
import shutil
import argparse
import numpy as np

from utils.config import Config


def load_labels_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    if ext == '.npz':
        d = np.load(path)
        # prefer 'labels' key if present
        if 'labels' in d.files:
            return d['labels']
        # otherwise return first array
        return d[d.files[0]]
    if ext in ('.csv', '.txt'):
        return np.loadtxt(path)
    if ext == '.ply':
        # try to use plyfile to extract scalar_seam or seam
        try:
            from plyfile import PlyData
            plydata = PlyData.read(path)
            if 'vertex' in plydata.elements:
                v = plydata['vertex'].data
                for candidate in ('scalar_seam', 'seam', 'label', 'labels'):
                    if candidate in v.dtype.names:
                        return np.array(v[candidate])
        except Exception as e:
            raise RuntimeError(f"Failed to read labels from PLY {path}: {e}")
    raise RuntimeError(f"Unsupported label file format: {path}")


def inject_labels(label_path, processed_dir, overwrite=False):
    stem = os.path.splitext(os.path.basename(label_path))[0]
    target_npz = os.path.join(processed_dir, stem + '.npz')
    if not os.path.exists(target_npz):
        print(f"No processed npz found for {stem} -> expected {target_npz}; skipping")
        return

    labels = load_labels_from_file(label_path)
    labels = np.asarray(labels)
    # flatten possible (N,1) into (N,)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)

    data = dict(np.load(target_npz))
    if 'features' not in data:
        print(f"No 'features' in {target_npz}; skipping")
        return
    if 'labels' in data and not overwrite:
        print(f"Target {target_npz} already contains 'labels' and --overwrite not set; skipping")
        return
    n_pts = data['features'].shape[0]
    if labels.shape[0] != n_pts:
        print(f"Label length {labels.shape[0]} != points {n_pts} for {stem}; skipping")
        return

    # ensure shape (N,1) float32
    labs = labels.astype(np.float32)
    if labs.ndim == 1:
        labs = labs[:, None]

    # backup original if not backed up
    backup = target_npz + '.orig'
    if not os.path.exists(backup):
        shutil.copy2(target_npz, backup)

    # merge and write
    data['labels'] = labs
    np.savez_compressed(target_npz, **data)
    print(f"Injected labels into {target_npz} (overwrite={overwrite})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing labels in npz')
    args = p.parse_args()

    label_dir = getattr(Config, 'LABEL_DATA_DIR', 'data/raw/label')
    processed_dir = getattr(Config, 'PROCESSED_DATA_DIR', 'data/processed/label')

    if not os.path.exists(label_dir):
        print(f"Label dir does not exist: {label_dir}")
        return
    if not os.path.exists(processed_dir):
        print(f"Processed dir does not exist: {processed_dir}")
        return

    supported = ('*.npy', '*.npz', '*.csv', '*.txt', '*.ply')
    files = []
    for pat in supported:
        files.extend(sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith(pat.lstrip('*'))]))

    if not files:
        print(f"No label files found in {label_dir}")
        return

    for f in files:
        try:
            inject_labels(f, processed_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"Failed to inject {f}: {e}")


if __name__ == '__main__':
    main()
