#!/usr/bin/env python3
"""Package the final 1/9/17 comparison clouds for local camera adjustment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/final_visualization_pointclouds_1_9_17_20260824"
SAMPLES = ("1 - 1Cloud_aug_18", "9 - Cloud_aug_18", "17 - Cloud_aug_18")
METHODS = ("GT", "DGCNN", "PTv3", "PointNet++", "PointMLP", "PointConv", "Ours")


def match_coordinate_units(xyz: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src_lo, src_hi = np.quantile(xyz, (0.001, 0.999), axis=0)
    ref_lo, ref_hi = np.quantile(reference, (0.001, 0.999), axis=0)
    scale = (ref_hi - ref_lo) / np.maximum(src_hi - src_lo, 1e-12)
    return ((xyz - src_lo) * scale + ref_lo).astype(np.float32)


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    xyz = df[["x", "y", "z"]].to_numpy(np.float64)
    label = (df["label"].to_numpy() > 0.5).astype(np.uint8)
    pred = (df["pred"].to_numpy() > 0.5).astype(np.uint8)
    return xyz, label, pred


def read_pointconv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :3], (arr[:, 3] > 0.5).astype(np.uint8), (arr[:, 4] > 0.5).astype(np.uint8)


def write_binary_ply(path: Path, xyz: np.ndarray, label: np.ndarray, pred: np.ndarray) -> None:
    colors = np.full((len(xyz), 3), 255, dtype=np.uint8)
    colors[pred > 0] = np.array([255, 0, 0], dtype=np.uint8)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "u1"), ("pred", "u1"),
    ])
    data = np.empty(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    data["label"], data["pred"] = label, pred
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar label\nproperty uchar pred\nend_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        data.tofile(handle)


def sources(stem: str) -> dict[str, tuple[Path, str]]:
    old = ROOT / "results/final_benchmark_78764/predictions"
    return {
        "DGCNN": (old / "dgcnn/seed444/fullpred_test" / f"{stem}_fullpred.csv", "csv"),
        "PTv3": (ROOT / "results/visualization/ptv3_fullpred" / f"{stem}_fullpred.csv", "csv"),
        "PointNet++": (ROOT / "results/pointnet2_fixed_78773/predictions/seed444/fullpred_test" / f"{stem}_fullpred.csv", "csv"),
        "PointMLP": (old / "pointmlp/seed444/fullpred_test" / f"{stem}_label_auto.csv", "csv"),
        "PointConv": (old / "pointconv/seed444/pred_xyz_label_test" / f"{stem}.txt", "txt"),
        "Ours": (old / "ours/seed444" / f"{stem}_label_auto_fullpred.csv", "csv"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"samples": list(SAMPLES), "methods": list(METHODS), "seed": 444, "files": []}
    data_root = ROOT / "comparable_methods/Pointcept-copy/data/weld_1_17/test"

    for stem in SAMPLES:
        raw_xyz = np.load(data_root / stem / "coord.npy").astype(np.float32)
        raw_label = (np.load(data_root / stem / "segment.npy").reshape(-1) > 0).astype(np.uint8)
        sample_dir = OUT / stem
        write_binary_ply(sample_dir / "00_GT.ply", raw_xyz, raw_label, raw_label)
        manifest["files"].append({"sample": stem, "method": "GT", "points": len(raw_xyz), "file": f"{stem}/00_GT.ply"})

        for index, method in enumerate(METHODS[1:], start=1):
            path, kind = sources(stem)[method]
            xyz, label, pred = read_pointconv(path) if kind == "txt" else read_csv(path)
            if not (len(xyz) == len(label) == len(pred) == len(raw_xyz)):
                raise ValueError(f"{method}/{stem}: point-count mismatch")
            xyz = match_coordinate_units(xyz, raw_xyz)
            filename = f"{index:02d}_{method.replace('+', 'plus')}.ply"
            write_binary_ply(sample_dir / filename, xyz, label, pred)
            manifest["files"].append({
                "sample": stem, "method": method, "points": len(xyz),
                "gt_positive": int(label.sum()), "pred_positive": int(pred.sum()),
                "source": str(path.relative_to(ROOT)), "file": f"{stem}/{filename}",
            })

    (OUT / "camera_presets.json").write_text(json.dumps({
        "coordinate_frame": "raw millimetre frame, axes aligned across methods",
        "matplotlib_reference_views": {
            "1 - 1Cloud_aug_18": {"elevation": 58, "azimuth": -62, "roll": -7},
            "9 - Cloud_aug_18": {"elevation": 38, "azimuth": -68, "roll": -2},
            "17 - Cloud_aug_18": {"elevation": 25, "azimuth": -90, "roll": 0},
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Final weld visualization point clouds (1, 9, 17)\n\n"
        "Open the binary PLY files in CloudCompare, MeshLab or Open3D. Each sample folder contains "
        "GT followed by DGCNN, PTv3, fixed PointNet++, PointMLP, PointConv and Ours. Coordinates are "
        "aligned to the same raw millimetre frame within each sample, so one saved camera can be reused "
        "for every method in that folder. RGB is white for predicted background and red for predicted weld. "
        "The PLY also contains `label` and `pred` scalar fields. Use a black viewer background.\n\n"
        "PTv3 Type 17 is the genuine archived successful-run output and predicts almost every point as weld.\n",
        encoding="utf-8",
    )
    print(f"Built {len(manifest['files'])} PLY files at {OUT}")


if __name__ == "__main__":
    main()
