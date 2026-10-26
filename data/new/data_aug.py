import os
import random
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Dict, Any

from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement


def load_ply(file_path: str) -> Dict[str, Any]:
    """加载PLY并保留额外字段（如label/label_soft/scalar_seam）。"""
    ply_data = PlyData.read(file_path)
    vertex = ply_data["vertex"]
    names = list(vertex.data.dtype.names or [])
    if not {"x", "y", "z"}.issubset(set(names)):
        raise ValueError(f"PLY 顶点缺少 x/y/z 字段，实际字段={names}")

    xyz = np.column_stack([
        vertex["x"].astype(np.float32),
        vertex["y"].astype(np.float32),
        vertex["z"].astype(np.float32),
    ]).astype(np.float32, copy=False)

    extra: Dict[str, np.ndarray] = {}
    for n in names:
        if n in ("x", "y", "z"):
            continue
        extra[n] = np.asarray(vertex[n])

    print(f"成功加载 {file_path}：{xyz.shape[0]} 个点，额外字段={list(extra.keys())}")
    return {"xyz": xyz, "extra": extra}


def save_ply(file_path: str, xyz: np.ndarray, extra: Dict[str, np.ndarray]):
    """保存PLY：x/y/z + 原始额外字段。"""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz 需要是 (N,3)，实际={xyz.shape}")

    n = xyz.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    cleaned_extra: Dict[str, np.ndarray] = {}
    for k, v in extra.items():
        arr = np.asarray(v)
        if arr.shape[0] != n:
            raise ValueError(f"字段 {k} 长度不一致: {arr.shape[0]} vs {n}")
        if k == "scalar_label":
            arr = np.nan_to_num(arr, nan=0.0)
        cleaned_extra[k] = arr
        dtype.append((k, arr.dtype.str))

    vertices = np.empty(n, dtype=dtype)
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    for k, v in cleaned_extra.items():
        vertices[k] = np.asarray(v)

    vertex_element = PlyElement.describe(vertices, "vertex")
    PlyData([vertex_element], text=True).write(file_path)
    print(f"已保存：{file_path}")


@dataclass
class AugConfig:
    num_augment: int = 10
    trans_range_xy: float = 1.0
    trans_range_z: float = 0.1
    trans_prob: float = 1.0
    rot_prob: float = 1.0
    rot_x: float = np.pi / 60
    rot_y: float = np.pi / 60
    rot_z: float = np.pi
    scale_min: float = 0.98
    scale_max: float = 1.02
    scale_prob: float = 0.7
    perturb_range_xy: float = 0.1
    perturb_range_z: float = 0.05
    perturb_prob: float = 0.4


def augment_xyz(xyz: np.ndarray, config: AugConfig) -> np.ndarray:
    """仅增强坐标，标签字段保持一一对应不变。"""
    xyz = np.asarray(xyz, dtype=np.float32).copy()

    if np.random.rand() < config.trans_prob:
        xyz += np.array([
            np.random.uniform(-config.trans_range_xy, config.trans_range_xy),
            np.random.uniform(-config.trans_range_xy, config.trans_range_xy),
            np.random.uniform(-config.trans_range_z, config.trans_range_z),
        ], dtype=np.float32)

    if np.random.rand() < config.rot_prob:
        rot = R.from_euler(
            "xyz",
            [
                np.random.uniform(-config.rot_x, config.rot_x),
                np.random.uniform(-config.rot_y, config.rot_y),
                np.random.uniform(-config.rot_z, config.rot_z),
            ],
        ).as_matrix().astype(np.float32)
        xyz = xyz @ rot.T

    if np.random.rand() < config.scale_prob:
        xyz = xyz * np.float32(np.random.uniform(config.scale_min, config.scale_max))

    if np.random.rand() < config.perturb_prob:
        noise = np.zeros_like(xyz)
        noise[:, 0] = np.random.uniform(-config.perturb_range_xy, config.perturb_range_xy, xyz.shape[0])
        noise[:, 1] = np.random.uniform(-config.perturb_range_xy, config.perturb_range_xy, xyz.shape[0])
        noise[:, 2] = np.random.uniform(-config.perturb_range_z, config.perturb_range_z, xyz.shape[0])
        xyz += noise.astype(np.float32)

    return xyz


def process_directory(input_dir: str, output_dir: str, config: AugConfig):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if not file.endswith(".ply"):
            continue
        input_path = os.path.join(input_dir, file)
        try:
            sample = load_ply(input_path)
        except Exception as e:
            print(f"跳过 {file}：{e}")
            continue

        xyz = sample["xyz"]
        extra = sample["extra"]
        base_name = os.path.splitext(file)[0]

        for i in range(config.num_augment):
            aug_xyz = augment_xyz(xyz, config)
            aug_extra = {k: (np.nan_to_num(v, nan=0.0) if k == "scalar_label" else v) for k, v in extra.items()}
            save_path = os.path.join(output_dir, f"{base_name}_aug_{i+1}.ply")
            save_ply(save_path, aug_xyz, aug_extra)

    print("所有文件处理完成！")


def _build_config_from_args(args) -> AugConfig:
    return AugConfig(
        num_augment=args.num_augment,
        trans_range_xy=args.trans_range_xy,
        trans_range_z=args.trans_range_z,
        trans_prob=args.trans_prob,
        rot_prob=args.rot_prob,
        rot_x=args.rot_x,
        rot_y=args.rot_y,
        rot_z=args.rot_z,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_prob=args.scale_prob,
        perturb_range_xy=args.perturb_range_xy,
        perturb_range_z=args.perturb_range_z,
        perturb_prob=args.perturb_prob,
    )


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="焊缝点云增强工具（支持保留label/soft label字段）")
    parser.add_argument("--input_dir", required=True, help="输入目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--num_augment", type=int, default=10)
    parser.add_argument("--trans_range_xy", type=float, default=1.0)
    parser.add_argument("--trans_range_z", type=float, default=0.1)
    parser.add_argument("--trans_prob", type=float, default=1.0)
    parser.add_argument("--rot_prob", type=float, default=1.0)
    parser.add_argument("--rot_x", type=float, default=float(np.pi / 60))
    parser.add_argument("--rot_y", type=float, default=float(np.pi / 60))
    parser.add_argument("--rot_z", type=float, default=float(np.pi))
    parser.add_argument("--scale_min", type=float, default=0.98)
    parser.add_argument("--scale_max", type=float, default=1.02)
    parser.add_argument("--scale_prob", type=float, default=0.7)
    parser.add_argument("--perturb_range_xy", type=float, default=0.1)
    parser.add_argument("--perturb_range_z", type=float, default=0.05)
    parser.add_argument("--perturb_prob", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证增强可复现")

    args = parser.parse_args()
    set_seed(args.seed)
    cfg = _build_config_from_args(args)
    process_directory(args.input_dir, args.output_dir, cfg)
