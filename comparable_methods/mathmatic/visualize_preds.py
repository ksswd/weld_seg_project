#!/usr/bin/env python3
"""
可视化数学方法输出的预测结果：
- 预测焊缝点：红色
- 其余点：白色（可选）

支持输入：
1) 单个 *_pred_method1.csv / *_pred_method2.csv 文件
2) 包含上述文件的目录
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_one(csv_path: str, out_dir: str, show_bg: bool, elev: float, azim: float, point_size: float) -> None:
    df = pd.read_csv(csv_path)
    required = {"x", "y", "z", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    pred = df["pred"].to_numpy(dtype=np.int32) > 0

    fig = plt.figure(figsize=(8, 6), dpi=180, facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    if show_bg:
        bg = ~pred
        if np.any(bg):
            ax.scatter(
                xyz[bg, 0], xyz[bg, 1], xyz[bg, 2],
                c="white", s=point_size, alpha=0.35, linewidths=0
            )

    if np.any(pred):
        ax.scatter(
            xyz[pred, 0], xyz[pred, 1], xyz[pred, 2],
            c="red", s=point_size * 1.2, alpha=0.95, linewidths=0
        )

    ax.view_init(elev=elev, azim=azim)

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    name = Path(csv_path).stem
    n_all = len(df)
    n_pos = int(pred.sum())
    ax.set_title(f"{name} | pred weld: {n_pos}/{n_all}", color="white")
    ax.set_axis_off()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="预测csv文件或目录")
    parser.add_argument("--out", default=None, help="输出目录")
    parser.add_argument("--show_bg", action="store_true", help="显示背景点")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--point_size", type=float, default=0.4)
    args = parser.parse_args()

    inp = args.input
    if os.path.isdir(inp):
        files = [
            os.path.join(inp, f)
            for f in sorted(os.listdir(inp))
            if f.endswith("_pred_method1.csv") or f.endswith("_pred_method2.csv")
        ]
        if not files:
            raise SystemExit(f"目录下未找到 *_pred_method1.csv / *_pred_method2.csv: {inp}")
        out_dir = args.out or os.path.join(inp, "vis")
    elif os.path.isfile(inp) and inp.endswith(".csv"):
        files = [inp]
        out_dir = args.out or os.path.dirname(inp)
    else:
        raise SystemExit("input 必须是预测csv文件或目录")

    print(f"将处理 {len(files)} 个文件，输出到: {out_dir}")
    for f in files:
        render_one(
            csv_path=f,
            out_dir=out_dir,
            show_bg=bool(args.show_bg),
            elev=float(args.elev),
            azim=float(args.azim),
            point_size=float(args.point_size),
        )


if __name__ == "__main__":
    main()
