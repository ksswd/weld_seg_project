#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def build_soft_labels(xyz: np.ndarray, hard: np.ndarray, radius: float) -> np.ndarray:
    hard = hard.astype(np.float32).reshape(-1)
    soft = hard.copy()

    pos_idx = np.where(hard >= 0.5)[0]
    neg_idx = np.where(hard < 0.5)[0]

    if len(pos_idx) == 0:
        return soft

    tree = cKDTree(xyz[pos_idx])
    dists, _ = tree.query(xyz[neg_idx], k=1, workers=-1)

    # y = 1 - d/r, d<=r; else 0
    cand = 1.0 - (dists / radius)
    cand = np.clip(cand, 0.0, 1.0)
    cand[dists > radius] = 0.0

    soft[neg_idx] = cand.astype(np.float32)
    soft[pos_idx] = 1.0
    return soft


def process_file(csv_path: str, out_path: str, radius: float):
    df = pd.read_csv(csv_path)
    required = {"x", "y", "z", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    hard = df["label"].to_numpy(dtype=np.float32)

    soft = build_soft_labels(xyz, hard, radius)
    df["label_soft"] = soft

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")

    pos = float((hard >= 0.5).mean())
    soft_pos = float((soft >= 0.5).mean())
    print(f"[ok] {os.path.basename(csv_path)} | hard_pos={pos:.4f} -> soft(>=0.5)={soft_pos:.4f} | saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="从label生成label_soft（基于最近焊缝点距离）")
    ap.add_argument("--input_dir", required=True, help="输入CSV目录（需包含x,y,z,label）")
    ap.add_argument("--output_dir", default=None, help="输出目录；不填则覆盖到输入目录")
    ap.add_argument("--radius", type=float, default=0.2, help="距离阈值，单位与点云一致（你的数据是mm）")
    ap.add_argument("--glob", default="*_label_*.csv", help="文件匹配模式")
    args = ap.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir or in_dir

    files = sorted(glob.glob(os.path.join(in_dir, args.glob)))
    if not files:
        raise SystemExit(f"未找到文件: {in_dir}/{args.glob}")

    print(f"found {len(files)} files, radius={args.radius}")
    for f in files:
        out_path = os.path.join(out_dir, os.path.basename(f))
        process_file(f, out_path, args.radius)


if __name__ == "__main__":
    main()
