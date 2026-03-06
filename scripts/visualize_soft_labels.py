#!/usr/bin/env python3
"""
可视化带 soft 标签的点云：
- 背景点（label_soft==0）：白色
- soft 点（label_soft>0）：按强度从浅红到深红渐变（值越大颜色越深）

示例：
python scripts/visualize_soft_labels.py --input_dir data/new/processed_csv_labeled_soft --out_dir data/new/processed_csv_labeled_soft/soft_vis
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_one(csv_path: str, out_dir: str, elev: float, azim: float, point_size: float):
    df = pd.read_csv(csv_path)
    need = {"x", "y", "z", "label_soft"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{csv_path} 缺少列: {miss}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    soft = df["label_soft"].to_numpy(dtype=np.float32)
    soft = np.clip(soft, 0.0, 1.0)

    bg = soft <= 1e-8
    fg = ~bg

    fig = plt.figure(figsize=(8, 6), dpi=180, facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # 背景：白色
    if np.any(bg):
        ax.scatter(
            xyz[bg, 0], xyz[bg, 1], xyz[bg, 2],
            c="white", s=point_size, alpha=0.28, linewidths=0
        )

    # soft点：按值映射红色深浅
    if np.any(fg):
        cmap = plt.cm.Reds
        # 避免太浅看不见，最低亮度从0.25起
        color_val = 0.25 + 0.75 * soft[fg]
        colors = cmap(color_val)
        ax.scatter(
            xyz[fg, 0], xyz[fg, 1], xyz[fg, 2],
            c=colors, s=point_size * 1.25, alpha=0.95, linewidths=0
        )

    # 统一尺度
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    name = Path(csv_path).stem
    n = len(df)
    n_soft = int((soft > 0).sum())
    ax.set_title(f"{name} | soft>0: {n_soft}/{n}", color="white", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{name}_soft.png")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="包含带label_soft列CSV的目录")
    ap.add_argument("--pattern", default="*_label_*.csv", help="匹配模式")
    ap.add_argument("--out_dir", default=None, help="输出目录，默认 input_dir/soft_vis")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-65.0)
    ap.add_argument("--point_size", type=float, default=0.35)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"未找到文件: {args.input_dir}/{args.pattern}")

    out_dir = args.out_dir or os.path.join(args.input_dir, "soft_vis")
    print(f"处理 {len(files)} 个文件 -> {out_dir}")

    for f in files:
        render_one(f, out_dir, args.elev, args.azim, args.point_size)


if __name__ == "__main__":
    main()
