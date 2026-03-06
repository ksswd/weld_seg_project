#!/usr/bin/env python3
"""
将硬标签 label(0/1) 批量转换为软标签 label_soft。

规则：
- 若 label == 1: label_soft = 1
- 若 label == 0:
    找最近的正样本点距离 d（基于 xyz，单位与数据一致，通常是 mm）
    当 d <= r 时: label_soft = 1 - d/r
    当 d >  r 时: label_soft = 0

用法示例：
python scripts/label_to_soft.py \
  --input_dir data/new/processed_csv_labeled \
  --glob "*_label_*.csv" \
  --radius 0.2 \
  --inplace

python scripts/label_to_soft.py \
  --input_dir data/new/processed_csv_labeled \
  --output_dir data/new/processed_csv_labeled_soft \
  --radius 0.2
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def build_soft_labels(df: pd.DataFrame, radius: float) -> np.ndarray:
    required_cols = {"x", "y", "z", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    hard = df["label"].to_numpy(dtype=np.float64)
    hard = np.where(np.isnan(hard), 0.0, hard)
    hard = (hard > 0.5).astype(np.float32)

    soft = np.zeros_like(hard, dtype=np.float32)
    pos_mask = hard == 1.0
    soft[pos_mask] = 1.0

    if pos_mask.sum() == 0:
        # 没有正样本，全部保持0
        return soft

    neg_mask = ~pos_mask
    if neg_mask.sum() == 0:
        return soft

    pos_xyz = xyz[pos_mask]
    neg_xyz = xyz[neg_mask]

    tree = cKDTree(pos_xyz)
    dists, _ = tree.query(neg_xyz, k=1)

    # 线性衰减：d<=r => 1-d/r；d>r => 0
    vals = 1.0 - (dists / float(radius))
    vals = np.clip(vals, 0.0, 1.0).astype(np.float32)
    soft[neg_mask] = vals

    return soft


def process_file(in_path: str, out_path: str, radius: float, overwrite: bool):
    df = pd.read_csv(in_path)

    if ("label_soft" in df.columns) and (not overwrite):
        print(f"[skip] {in_path} 已存在 label_soft（使用 --overwrite 覆盖）")
        return

    soft = build_soft_labels(df, radius=radius)
    df["label_soft"] = soft

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")

    pos_ratio = float((df["label"] > 0.5).mean()) if "label" in df.columns else float("nan")
    soft_mean = float(df["label_soft"].mean())
    soft_gt0 = float((df["label_soft"] > 0).mean())
    print(
        f"[ok] {os.path.basename(in_path)} -> {out_path} | "
        f"pos_ratio={pos_ratio:.4f}, soft_mean={soft_mean:.4f}, soft>0 ratio={soft_gt0:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="输入CSV目录（包含 label 列）")
    ap.add_argument("--glob", default="*_label_*.csv", help="文件匹配模式")
    ap.add_argument("--output_dir", default=None, help="输出目录；不填且 --inplace 时原地写回")
    ap.add_argument("--radius", type=float, default=0.2, help="软标签半径（单位同xyz）")
    ap.add_argument("--inplace", action="store_true", help="原地覆盖写入")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有 label_soft 列")
    args = ap.parse_args()

    if args.radius <= 0:
        raise ValueError("--radius 必须 > 0")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    if not files:
        raise FileNotFoundError(f"未找到文件: {args.input_dir}/{args.glob}")

    if not args.inplace and args.output_dir is None:
        # 默认安全模式：输出到 input_dir_soft
        args.output_dir = str(Path(args.input_dir).with_name(Path(args.input_dir).name + "_soft"))

    for in_path in files:
        if args.inplace:
            out_path = in_path
        else:
            out_path = os.path.join(args.output_dir, os.path.basename(in_path))
        process_file(in_path, out_path, radius=args.radius, overwrite=args.overwrite)

    print(f"完成，共处理 {len(files)} 个文件。")


if __name__ == "__main__":
    main()
