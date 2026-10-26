#!/usr/bin/env python3
"""将 fullpred CSV 转为带概率灰度着色的 PLY 点云。

输入 CSV 需要至少包含列：x, y, z, prob
输出 PLY 使用 RGB 颜色表示概率：
- 概率越高，颜色越深（越黑）
- 概率越低，颜色越浅（越白）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import open3d as o3d


def prob_to_gray_rgb(prob: np.ndarray) -> np.ndarray:
    """将概率映射到灰度 RGB，概率越高越黑。"""
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    prob = np.clip(prob, 0.0, 1.0)
    gray = (255.0 * (1.0 - prob)).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=1)


def csv_to_ply(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)

    required_cols = {"x", "y", "z", "prob"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列: {sorted(missing)}，当前列: {list(df.columns)}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    prob = df["prob"].to_numpy(dtype=np.float32)
    colors = prob_to_gray_rgb(prob).astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=True):
        raise RuntimeError(f"写入 PLY 失败: {out_path}")


def collect_csv_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
    else:
        yield from sorted(input_path.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="将 fullpred CSV 转换为按概率灰度着色的 PLY")
    parser.add_argument("--input", type=str, required=True, help="单个 CSV 文件或包含 CSV 的目录")
    parser.add_argument("--output", type=str, default=None, help="输出 PLY 文件或输出目录")
    parser.add_argument("--suffix", type=str, default="_prob", help="批量转换时输出文件后缀")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if args.output is None:
            out_path = input_path.with_suffix(".ply")
        else:
            out_path = Path(args.output)
            if out_path.is_dir() or str(args.output).endswith(("/", "\\")):
                out_path = out_path / f"{input_path.stem}{args.suffix}.ply"
        csv_to_ply(input_path, out_path)
        print(f"已转换: {input_path} -> {out_path}")
    else:
        out_dir = Path(args.output) if args.output is not None else input_path
        out_dir.mkdir(parents=True, exist_ok=True)
        files = list(collect_csv_files(input_path))
        if not files:
            raise FileNotFoundError(f"目录中未找到 CSV 文件: {input_path}")
        for csv_path in files:
            out_path = out_dir / f"{csv_path.stem}{args.suffix}.ply"
            csv_to_ply(csv_path, out_path)
            print(f"已转换: {csv_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
