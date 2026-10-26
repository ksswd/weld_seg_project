#!/usr/bin/env python3
"""Build standardized point-cloud predictions and PNGs for all benchmark methods."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEARNING = {
    "ours": ("results/unified_inference/ours/ablation_geom_on_gate_on_seed444_pts4096", "{name}_label_auto_fullpred.csv", "pred"),
    "dgcnn": ("comparable_methods/dgcnn.pytorch/outputs/unified_inference/dgcnn_seed444_pts4096/fullpred_test", "{name}_fullpred.csv", "pred"),
    "pointnet2": ("comparable_methods/Pointnet_Pointnet2_pytorch/fullpred/unified_inference/pointnet2_seed444_pts4096/fullpred_test", "{name}_fullpred.csv", "pred"),
    "pointmlp": ("comparable_methods/pointMLP-pytorch/part_segmentation/outputs/unified_inference/pointmlp_seed444_pts4096/fullpred_test", "{name}_label_auto.csv", "pred"),
    "pointconv": ("comparable_methods/pointconv_pytorch/log/seed_points_sweep/pointconv_seed444_pts4096/pred_xyz_label_test", "{name}.txt", 4),
    "point_transformer_v3": ("comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs/result", "{name}_pred.npy", None),
}


def load_math_module(root: Path):
    path = root / "comparable_methods/mathmatic/run_traditional_baselines.py"
    spec = importlib.util.spec_from_file_location("traditional_baselines", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save_visual(path: Path, xyz: np.ndarray, label: np.ndarray, pred: np.ndarray, seed: int, max_points: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(xyz))
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, max_points, replace=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, values, title in zip(axes, (label, pred), ("Ground truth", "Prediction")):
        ax.scatter(xyz[idx, 0], xyz[idx, 1], c=values[idx], s=0.8, cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_one(out: Path, method: str, name: str, xyz, label, pred, seed: int, max_points: int):
    target = out / method
    (target / "pointcloud_npz").mkdir(parents=True, exist_ok=True)
    (target / "visualizations").mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    label = np.asarray(label, dtype=np.uint8).reshape(-1)
    pred = np.asarray(pred, dtype=np.uint8).reshape(-1)
    if not (len(xyz) == len(label) == len(pred)):
        raise ValueError(f"{method}/{name}: length mismatch {len(xyz)}/{len(label)}/{len(pred)}")
    np.savez_compressed(target / "pointcloud_npz" / f"{name}.npz", xyz=xyz, label=label, pred=pred)
    save_visual(target / "visualizations" / f"{name}.png", xyz, label, pred, seed, max_points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/weld_comparison_visualizations_20260809")
    parser.add_argument("--max-vis-points", type=int, default=40000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    data_root = root / "comparable_methods/Pointcept-copy/data/weld_1_17/test"
    processed_root = root / "data/dataset_processed_1-17/test"
    samples = sorted(p.name for p in data_root.iterdir() if p.is_dir())

    manifest = {"representative_seed": 444, "num_test_clouds": len(samples), "methods": {}}
    for method, (directory, pattern, pred_col) in LEARNING.items():
        source = root / directory
        for i, name in enumerate(samples):
            xyz = np.load(data_root / name / "coord.npy")
            label = np.load(data_root / name / "segment.npy")
            path = source / pattern.format(name=name)
            if path.suffix == ".npy":
                pred = np.load(path)
            elif path.suffix == ".txt":
                pred = np.loadtxt(path, usecols=[pred_col])
            else:
                pred = pd.read_csv(path, usecols=[pred_col])[pred_col].to_numpy()
            export_one(out, method, name, xyz, label, pred, 444 + i, args.max_vis_points)
        manifest["methods"][method] = {"category": "learning", "source": directory}

    math = load_math_module(root)
    configs = json.loads((root / "results/traditional_baselines_optimized/selected_parameters.json").read_text())
    math_names = list(configs)
    for i, name in enumerate(samples):
        csv_path = processed_root / f"{name}_label_auto.csv"
        df = pd.read_csv(csv_path)
        xyz = np.load(data_root / name / "coord.npy")
        label = np.load(data_root / name / "segment.npy")
        _, _, _, r1, r2 = math.geometry_cache(csv_path, 120, 20000, 10042 + i * 2)
        for method in math_names:
            cfg = configs[method]
            if method.startswith("threshold_"):
                values = df[cfg["feature"]].to_numpy()
                pred = values >= cfg["threshold"] if cfg["direction"] == "ge" else values <= cfg["threshold"]
            elif method == "multi_feature_geometric_score":
                values = df[cfg["features"]].replace([np.inf, -np.inf], 0).to_numpy(dtype=np.float64)
                phi = math.polynomial_geometry_features(values)
                pred = phi @ np.asarray(cfg["coefficients"]) + cfg["intercept"] >= cfg["threshold"]
            elif method == "ransac_single_plane_residual":
                pred = r1 > cfg["distance_ratio"]
            else:
                pred = (r1 <= cfg["corridor_ratio"]) & (r2 <= cfg["corridor_ratio"])
            export_one(out, method, name, xyz, label, pred, 10442 + i, args.max_vis_points)
        
    for method in math_names:
        manifest["methods"][method] = {"category": "mathematical", "parameters": configs[method]}

    extras = {
        root / "results/unified_benchmark/benchmark_summary.csv": out / "tables/learning_methods_summary.csv",
        root / "results/traditional_baselines_optimized/traditional_baselines_summary.csv": out / "tables/mathematical_methods_summary.csv",
        root / "results/model_parameters.csv": out / "tables/model_parameters.csv",
        root / "docs/baseline_methods_and_results.md": out / "documentation/baseline_methods_and_results.md",
    }
    for source, target in extras.items():
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Weld segmentation visualization package\n\n"
        "Each method contains 57 `pointcloud_npz` files and 57 paired ground-truth/prediction PNGs. "
        "NPZ keys are `xyz` (N x 3 float32), `label` (N uint8), and `pred` (N uint8). "
        "Four-seed learning methods use seed 444 as the representative visualization run; "
        "Point Transformer V3 has one available checkpoint. Mathematical predictions use parameters "
        "selected only on the validation split. See `manifest.json` for provenance.\n",
        encoding="utf-8",
    )
    print(f"Built {len(manifest['methods'])} methods x {len(samples)} clouds at {out}")


if __name__ == "__main__":
    main()
