#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def read_ascii_ply_xyz_label(ply_path: Path):
    with ply_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{ply_path} 不是合法 PLY 文件")

    vertex_count = None
    prop_names = []
    header_end = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            vertex_count = int(s.split()[-1])
        elif s.startswith("property"):
            parts = s.split()
            if len(parts) >= 3:
                prop_names.append(parts[-1])
        elif s == "end_header":
            header_end = i
            break

    if header_end is None:
        raise ValueError(f"{ply_path} 缺少 end_header")
    if vertex_count is None:
        raise ValueError(f"{ply_path} 缺少 element vertex")

    required = ["x", "y", "z", "scalar_label"]
    for k in required:
        if k not in prop_names:
            raise ValueError(f"{ply_path} 缺少属性: {k}, 当前属性={prop_names}")

    data_lines = lines[header_end + 1 : header_end + 1 + vertex_count]
    if len(data_lines) < vertex_count:
        raise ValueError(f"{ply_path} 点数不足，header={vertex_count}, 实际={len(data_lines)}")

    arr = np.loadtxt(data_lines, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != len(prop_names):
        raise ValueError(
            f"{ply_path} 列数不匹配，header属性数={len(prop_names)}, 数据列数={arr.shape[1]}"
        )

    idx = {name: i for i, name in enumerate(prop_names)}
    coord = arr[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    segment = arr[:, idx["scalar_label"]].astype(np.int64)
    return coord, segment


def save_pointcept_sample(sample_out_dir: Path, coord: np.ndarray, segment: np.ndarray):
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    n = coord.shape[0]
    normal = np.zeros((n, 3), dtype=np.float32)
    color = np.zeros((n, 3), dtype=np.float32)

    np.save(sample_out_dir / "coord.npy", coord)
    np.save(sample_out_dir / "normal.npy", normal)
    np.save(sample_out_dir / "color.npy", color)
    np.save(sample_out_dir / "segment.npy", segment)


def convert_split(src_split_dir: Path, dst_split_dir: Path):
    ply_files = sorted(src_split_dir.glob("*.ply"))
    print(f"[{src_split_dir.name}] 找到 {len(ply_files)} 个 ply 文件")

    converted = 0
    for ply_path in ply_files:
        coord, segment = read_ascii_ply_xyz_label(ply_path)
        sample_name = ply_path.stem
        out_dir = dst_split_dir / sample_name
        save_pointcept_sample(out_dir, coord, segment)
        converted += 1

    print(f"[{src_split_dir.name}] 转换完成: {converted}")


def main():
    parser = argparse.ArgumentParser(description="将 train/validation/test 的 ply 数据转换为 Pointcept 格式")
    parser.add_argument(
        "--in-root",
        type=str,
        default="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/dataset",
        help="输入数据根目录，包含 train/validation/test",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/weld_dataset",
        help="输出 Pointcept 数据根目录",
    )
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for src_name, dst_name in split_map.items():
        src_dir = in_root / src_name
        if not src_dir.exists():
            raise FileNotFoundError(f"未找到输入目录: {src_dir}")
        dst_dir = out_root / dst_name
        convert_split(src_dir, dst_dir)

    print("全部转换完成")
    print(f"输出目录: {out_root.resolve()}")


if __name__ == "__main__":
    main()
