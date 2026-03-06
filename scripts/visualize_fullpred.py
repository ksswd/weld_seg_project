#!/usr/bin/env python3
"""
将 full_eval 输出的 *_fullpred.csv 渲染为固定视角图片：
- 预测为焊缝的点：红色
- 其余点：白色（可选不显示）

用法示例：
python scripts/visualize_fullpred.py data/new/full_eval_all --out data/new/full_eval_all/vis --thr 0.53 --show_bg
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_one(csv_path: str, out_dir: str, thr: float, show_bg: bool, elev: float, azim: float, point_size: float):
    df = pd.read_csv(csv_path)
    required = {"x", "y", "z", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    pred = (df["prob"].to_numpy(dtype=np.float32) > thr)

    # 黑色背景
    fig = plt.figure(figsize=(8, 6), dpi=180, facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # 背景点（非焊缝）
    if show_bg:
        bg = ~pred
        if np.any(bg):
            ax.scatter(
                xyz[bg, 0], xyz[bg, 1], xyz[bg, 2],
                c="white", s=point_size, alpha=0.35, linewidths=0
            )

    # 预测焊缝点（红色）
    if np.any(pred):
        ax.scatter(
            xyz[pred, 0], xyz[pred, 1], xyz[pred, 2],
            c="red", s=point_size * 1.2, alpha=0.95, linewidths=0
        )

    # 固定视角
    ax.view_init(elev=elev, azim=azim)

    # 统一轴范围，避免拉伸
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    # 文字改白
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="z", colors="white")
    ax.xaxis.pane.set_facecolor((0, 0, 0, 1))
    ax.yaxis.pane.set_facecolor((0, 0, 0, 1))
    ax.zaxis.pane.set_facecolor((0, 0, 0, 1))
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")

    name = Path(csv_path).stem.replace("_fullpred", "")
    n_all = len(df)
    n_pos = int(pred.sum())
    ax.set_title(f"{name} | pred weld: {n_pos}/{n_all} | thr={thr:.3f}", color="white")

    # 去掉坐标轴与边框
    ax.set_axis_off()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{name}_pred.png")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="一个 *_fullpred.csv 文件或包含它们的目录")
    ap.add_argument("--out", default=None, help="输出目录，默认与输入同级/同目录")
    ap.add_argument("--thr", type=float, default=0.53, help="预测阈值")
    ap.add_argument("--show_bg", action="store_true", help="显示非焊缝背景点（白色）")
    ap.add_argument("--elev", type=float, default=20.0, help="视角 elev")
    ap.add_argument("--azim", type=float, default=-65.0, help="视角 azim")
    ap.add_argument("--point_size", type=float, default=0.4, help="点大小")
    args = ap.parse_args()

    inp = args.input
    if os.path.isdir(inp):
        files = [
            os.path.join(inp, f)
            for f in sorted(os.listdir(inp))
            if f.endswith("_fullpred.csv")
        ]
        if not files:
            raise SystemExit(f"目录下未找到 *_fullpred.csv: {inp}")
        out_dir = args.out or os.path.join(inp, "vis")
    elif os.path.isfile(inp) and inp.endswith(".csv"):
        files = [inp]
        out_dir = args.out or os.path.dirname(inp)
    else:
        raise SystemExit("input 必须是 *_fullpred.csv 文件或包含该文件的目录")

    print(f"将处理 {len(files)} 个文件，输出到: {out_dir}")
    for f in files:
        render_one(
            csv_path=f,
            out_dir=out_dir,
            thr=float(args.thr),
            show_bg=bool(args.show_bg),
            elev=float(args.elev),
            azim=float(args.azim),
            point_size=float(args.point_size),
        )


if __name__ == "__main__":
    main()
