#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def robust_zscore(x, eps=1e-6):
    x = np.asarray(x).reshape(-1, 1)
    med = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    if iqr < eps:
        iqr = eps
    return ((x.reshape(-1) - med) / iqr).astype(np.float32)


def minmax_clip(x, p_low=1, p_high=99, eps=1e-6):
    x = np.asarray(x).reshape(-1)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi - lo < eps:
        return np.zeros_like(x)
    y = np.clip(x, lo, hi)
    return (y - lo) / (hi - lo)


def knn_indices(xyz, k):
    k = min(k, len(xyz))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    return distances, indices


def pca_features_from_knn(xyz, knn_idx):
    n = xyz.shape[0]
    curvature = np.zeros(n, dtype=np.float32)
    linearity = np.zeros(n, dtype=np.float32)
    planarity = np.zeros(n, dtype=np.float32)
    scattering = np.zeros(n, dtype=np.float32)
    eps = 1e-12

    for i in range(n):
        pts = xyz[knn_idx[i]]
        pts_centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (pts_centered.T @ pts_centered) / max(len(pts) - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0.0))
        l0, l1, l2 = eigvals[0], eigvals[1], eigvals[2]
        s = l0 + l1 + l2 + eps
        curvature[i] = l0 / s
        linearity[i] = (l2 - l1) / (l2 + eps)
        planarity[i] = (l1 - l0) / (l2 + eps)
        scattering[i] = l0 / (l2 + eps)

    return {"curvature": curvature, "linearity": linearity, "planarity": planarity, "scattering": scattering}


def median_filter_feature(values, knn_idx):
    values = np.asarray(values).reshape(-1)
    out = np.zeros_like(values, dtype=np.float32)
    for i in range(len(values)):
        out[i] = np.median(values[knn_idx[i]])
    return out


def compute_knn_spacing_density(distances, k_density):
    eps = 1e-8
    kth = min(k_density - 1, distances.shape[1] - 1)
    spacing = distances[:, kth].astype(np.float32)
    density = np.log(1.0 / (spacing + eps)).astype(np.float32)
    return spacing, density


def plot_three_views(xyz, values, title, cmap="viridis", out_path=None, max_points=None, value_label="value", p_low=1, p_high=99):
    values = np.asarray(values).reshape(-1)
    if max_points is not None and len(xyz) > max_points:
        idx = np.random.choice(len(xyz), size=max_points, replace=False)
        xyz_plot = xyz[idx]
        val_plot = values[idx]
    else:
        xyz_plot = xyz
        val_plot = values

    val_vis = minmax_clip(val_plot, p_low=p_low, p_high=p_high)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    views = [("XY view", 0, 1, "x", "y"), ("XZ view", 0, 2, "x", "z"), ("YZ view", 1, 2, "y", "z")]
    sc = None
    for ax, (view_title, a, b, xlabel, ylabel) in zip(axes, views):
        sc = ax.scatter(xyz_plot[:, a], xyz_plot[:, b], c=val_vis, s=1, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(view_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title, fontsize=14)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label(value_label)
    if out_path is not None:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one(csv_path: Path, out_csv_path: Path, keep_new_columns: bool = True, replace_original_columns: bool = True, vis_dir: Path | None = None, vis_max_points: int = 30000):
    K_MEDIAN_CURV = 40
    K_PCA_SMALL = 30
    K_PCA_MID = 80
    K_PCA_LARGE = 150
    K_DENSITY = 50
    TEXTURE_SUPPRESS_ALPHA = 0.5

    df = pd.read_csv(csv_path)
    for c in ["x", "y", "z"]:
    if c not in df.columns:
        raise ValueError(f"CSV 缺少必要列: {c}")

xyz = df[["x", "y", "z"]].values.astype(np.float32)
K_MAX = max(K_MEDIAN_CURV, K_PCA_SMALL, K_PCA_MID, K_PCA_LARGE, K_DENSITY)
distances, indices = knn_indices(xyz, K_MAX)
    idx_median = indices[:, : min(K_MEDIAN_CURV, indices.shape[1])]
    idx_small = indices[:, : min(K_PCA_SMALL, indices.shape[1])]
    idx_mid = indices[:, : min(K_PCA_MID, indices.shape[1])]
    idx_large = indices[:, : min(K_PCA_LARGE, indices.shape[1])]

if "curvature" in df.columns:
    curv_raw = df["curvature"].values.astype(np.float32)
elif "curvature_norm" in df.columns:
    curv_raw = df["curvature_norm"].values.astype(np.float32)
else:
    raise ValueError("CSV 中找不到 curvature 或 curvature_norm 列")

curv_median = median_filter_feature(curv_raw, idx_median)
feat_small = pca_features_from_knn(xyz, idx_small)
feat_mid = pca_features_from_knn(xyz, idx_mid)
feat_large = pca_features_from_knn(xyz, idx_large)
curv_small = feat_small["curvature"]
curv_mid = feat_mid["curvature"]
curv_large = feat_large["curvature"]
texture_response = np.maximum(0.0, curv_small - curv_large)
    curv_clean = np.maximum(curv_mid - TEXTURE_SUPPRESS_ALPHA * texture_response, 0.0)
curv_clean = median_filter_feature(curv_clean, idx_median)
curv_clean_norm = robust_zscore(curv_clean)

if "linearity" in df.columns:
    linearity_raw = df["linearity"].values.astype(np.float32)
else:
    linearity_raw = np.zeros(len(df), dtype=np.float32)
    linearity_clean = median_filter_feature(feat_mid["linearity"], idx_median)
linearity_clean = np.clip(linearity_clean, 0.0, 1.0).astype(np.float32)

if "density" in df.columns:
    density_raw = df["density"].values.astype(np.float32)
elif "density_norm" in df.columns:
    density_raw = df["density_norm"].values.astype(np.float32)
elif "local_density" in df.columns:
    density_raw = df["local_density"].values.astype(np.float32)
else:
    density_raw = np.zeros(len(df), dtype=np.float32)

spacing, density_knn = compute_knn_spacing_density(distances, K_DENSITY)
density_clean_norm = robust_zscore(density_knn)
spacing_norm = robust_zscore(spacing)

df_out = df.copy()
    if keep_new_columns:
    df_out["curvature_raw_before_denoise"] = curv_raw
    df_out["curvature_median"] = curv_median
    df_out["curvature_pca_small"] = curv_small
    df_out["curvature_pca_mid"] = curv_mid
    df_out["curvature_pca_large"] = curv_large
    df_out["curvature_texture_response"] = texture_response
    df_out["curvature_clean"] = curv_clean
    df_out["curvature_clean_norm"] = curv_clean_norm
    df_out["linearity_raw_before_denoise"] = linearity_raw
    df_out["linearity_clean"] = linearity_clean
    df_out["density_raw_before_denoise"] = density_raw
    df_out["density_knn"] = density_knn
    df_out["density_clean_norm"] = density_clean_norm
    df_out["spacing_knn"] = spacing
    df_out["spacing_knn_norm"] = spacing_norm

    if replace_original_columns:
    if "curvature" in df_out.columns:
        df_out["curvature"] = curv_clean_norm
    elif "curvature_norm" in df_out.columns:
        df_out["curvature_norm"] = curv_clean_norm
    if "density" in df_out.columns:
        df_out["density"] = density_clean_norm
    elif "density_norm" in df_out.columns:
        df_out["density_norm"] = density_clean_norm
    elif "local_density" in df_out.columns:
        df_out["local_density"] = density_clean_norm
        df_out["linearity"] = linearity_clean

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv_path, index=False)
    print(f"Saved: {out_csv_path}")

    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        base_name = csv_path.stem
        plot_three_views(xyz, curv_raw, f"{base_name} - curvature raw", out_path=vis_dir / f"{base_name}_curvature_raw.png", cmap="viridis", max_points=vis_max_points, value_label="curvature")
        plot_three_views(xyz, curv_clean_norm, f"{base_name} - curvature clean", out_path=vis_dir / f"{base_name}_curvature_clean.png", cmap="viridis", max_points=vis_max_points, value_label="curvature")
        plot_three_views(xyz, linearity_clean, f"{base_name} - linearity clean", out_path=vis_dir / f"{base_name}_linearity_clean.png", cmap="viridis", max_points=vis_max_points, value_label="linearity")
        plot_three_views(xyz, density_clean_norm, f"{base_name} - density clean", out_path=vis_dir / f"{base_name}_density_clean.png", cmap="viridis", max_points=vis_max_points, value_label="density")


def collect_csv_files(root_dir: Path):
    return sorted([p for p in root_dir.rglob("*.csv") if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description="构建 fixed 数据集：对 17/18/19 号数据做特征修复，其余直接复制")
    parser.add_argument("--input_root", type=str, default="data/dataset_processed")
    parser.add_argument("--output_root", type=str, default="data/dataset_processed_fixed")
    parser.add_argument("--fix_ids", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_max_points", type=int, default=30000)
    parser.add_argument("--keep_new_columns", action="store_true")
    parser.add_argument("--replace_original_columns", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    fix_ids = {int(x) for x in args.fix_ids}

    csv_files = collect_csv_files(input_root)
    if not csv_files:
        raise FileNotFoundError(f"未找到 CSV: {input_root}")

    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"修复编号: {sorted(fix_ids)}")
    print(f"找到 {len(csv_files)} 个 CSV 文件")

    for csv_path in csv_files:
        rel_path = csv_path.relative_to(input_root)
        out_csv_path = output_root / rel_path
        file_id = csv_path.name.split(" ")[0]
        should_fix = file_id.isdigit() and int(file_id) in fix_ids

        if should_fix:
            vis_dir = out_csv_path.parent / "feature_denoise_vis" if args.vis else None
            run_one(
                csv_path=csv_path,
                out_csv_path=out_csv_path,
                keep_new_columns=args.keep_new_columns,
                replace_original_columns=args.replace_original_columns,
                vis_dir=vis_dir,
                vis_max_points=args.vis_max_points,
            )
        else:
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(csv_path)
            df.to_csv(out_csv_path, index=False)
            print(f"Copied: {csv_path} -> {out_csv_path}")


if __name__ == "__main__":
    main()
