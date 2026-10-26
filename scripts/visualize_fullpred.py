#!/usr/bin/env python3
"""
将多种预测输出渲染为固定视角图片：
- 预测为焊缝的点：红色
- 其余点：白色（可选不显示）

支持输入类型：
1) fullpred csv: 列含 x,y,z,prob（可选 pred）
2) 普通 csv: 列含 x,y,z,pred（无 prob）
3) txt: 无表头，按空白分隔，前5列视为 x,y,z,gt,pred
4) npy: 仅 pred 向量（需通过 --raw_xyz_dir 提供同名 csv 获取 xyz）

用法示例：
python scripts/visualize_fullpred.py data/new/full_eval_all --out data/new/full_eval_all/vis --thr 0.53 --show_bg
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_prediction_file(path: str, thr: float, raw_xyz_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        cols = set(df.columns)
        required_xyz = {"x", "y", "z"}
        if not required_xyz.issubset(cols):
            raise ValueError(f"{path} 缺少坐标列 x,y,z")
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

        if "prob" in cols:
            pred = df["prob"].to_numpy(dtype=np.float32) > float(thr)
        elif "pred" in cols:
            pred = df["pred"].to_numpy(dtype=np.float32) > 0.5
        elif "prediction" in cols:
            pred = df["prediction"].to_numpy(dtype=np.float32) > 0.5
        else:
            raise ValueError(f"{path} 既没有 prob/pred/prediction 列")
        return xyz, pred

    if suffix == ".txt":
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 5:
            raise ValueError(f"{path} txt 列数不足，期望至少 5 列（x y z gt pred）")
        xyz = arr[:, :3].astype(np.float32)
        pred = arr[:, 4] > 0.5
        return xyz, pred

    if suffix == ".npy":
        pred_arr = np.load(path)
        if pred_arr.ndim != 1:
            pred_arr = pred_arr.reshape(-1)
        pred = pred_arr.astype(np.float32) > 0.5

        if not raw_xyz_dir:
            raise ValueError(f"{path} 为 npy，仅包含 pred；请提供 --raw_xyz_dir 用于读取 xyz")

        stem = p.stem
        if stem.endswith("_pred"):
            stem = stem[:-5]
        raw_csv = Path(raw_xyz_dir) / f"{stem}.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"npy 对应原始坐标文件不存在: {raw_csv}")

        raw_df = pd.read_csv(raw_csv)
        required_xyz = {"x", "y", "z"}
        if not required_xyz.issubset(set(raw_df.columns)):
            raise ValueError(f"{raw_csv} 缺少坐标列 x,y,z")
        xyz = raw_df[["x", "y", "z"]].to_numpy(dtype=np.float32)

        if len(xyz) != len(pred):
            raise ValueError(
                f"长度不匹配: {path} pred={len(pred)} vs {raw_csv} xyz={len(xyz)}"
            )
        return xyz, pred

    raise ValueError(f"不支持的文件类型: {path}")


def render_one(
    pred_path: str,
    out_dir: str,
    thr: float,
    show_bg: bool,
    elev: float,
    azim: float,
    point_size: float,
    raw_xyz_dir: Optional[str],
) -> None:
    xyz, pred = _load_prediction_file(pred_path, thr=thr, raw_xyz_dir=raw_xyz_dir)

    fig = plt.figure(figsize=(8, 6), dpi=180, facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    if show_bg:
        bg = ~pred
        if np.any(bg):
            ax.scatter(
                xyz[bg, 0],
                xyz[bg, 1],
                xyz[bg, 2],
                c="white",
                s=point_size,
                alpha=0.35,
                linewidths=0,
            )

    if np.any(pred):
        ax.scatter(
            xyz[pred, 0],
            xyz[pred, 1],
            xyz[pred, 2],
            c="red",
            s=point_size * 1.2,
            alpha=0.95,
            linewidths=0,
        )

    ax.view_init(elev=elev, azim=azim)

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

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

    name = Path(pred_path).stem.replace("_fullpred", "").replace("_pred", "")
    n_all = int(len(pred))
    n_pos = int(pred.sum())
    ax.set_title(f"{name} | pred weld: {n_pos}/{n_all} | thr={thr:.3f}", color="white")

    ax.set_axis_off()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{name}_pred.png")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="单个文件（csv/txt/npy）或目录")
    ap.add_argument("--out", default=None, help="输出目录，默认 input/visualization")
    ap.add_argument("--thr", type=float, default=0.53, help="当有 prob 列时使用的阈值")
    ap.add_argument("--show_bg", action="store_true", help="显示非焊缝背景点（白色）")
    ap.add_argument("--elev", type=float, default=20.0, help="视角 elev")
    ap.add_argument("--azim", type=float, default=-65.0, help="视角 azim")
    ap.add_argument("--point_size", type=float, default=0.4, help="点大小")
    ap.add_argument("--raw_xyz_dir", default=None, help="当输入为 npy 时，同名原始 csv 所在目录")
    args = ap.parse_args()

    inp = args.input
    valid_suffix = (".csv", ".txt", ".npy")

    if os.path.isdir(inp):
        all_files = [
            os.path.join(inp, f)
            for f in sorted(os.listdir(inp))
            if f.lower().endswith(valid_suffix)
        ]
        # 目录模式下默认只可视化“预测结果文件”，避免把 metrics/summary 这类 CSV 当作点云读入
        files = []
        for p in all_files:
            name = os.path.basename(p).lower()
            if name.endswith(".csv"):
                if ("_fullpred" in name) or ("_pred_simple" in name) or name.endswith("_pred.csv"):
                    files.append(p)
            else:
                # txt / npy 保持原有行为
                files.append(p)
        if not files:
            raise SystemExit(f"目录下未找到可处理的预测文件: {inp}")
        out_dir = args.out or os.path.join(inp, "visualization")
    elif os.path.isfile(inp) and inp.lower().endswith(valid_suffix):
        files = [inp]
        out_dir = args.out or os.path.join(os.path.dirname(inp), "visualization")
    else:
        raise SystemExit("input 必须是 csv/txt/npy 文件或包含这些文件的目录")

    print(f"将处理 {len(files)} 个文件，输出到: {out_dir}")
    for f in files:
        render_one(
            pred_path=f,
            out_dir=out_dir,
            thr=float(args.thr),
            show_bg=bool(args.show_bg),
            elev=float(args.elev),
            azim=float(args.azim),
            point_size=float(args.point_size),
            raw_xyz_dir=args.raw_xyz_dir,
        )


if __name__ == "__main__":
    main()
