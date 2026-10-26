#!/usr/bin/env python3
"""Build objective best/worst mathematical-baseline case studies versus Ours."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparable_methods.mathmatic.run_traditional_baselines_v2 import (  # noqa: E402
    cloud_spacing,
    fit_dominant_planes,
    raw_path_for,
)
from comparable_methods.mathmatic.sweep_traditional_thresholds import (  # noqa: E402
    confusion,
    metrics,
    percentile_rank,
)
from utils.io_utils import read_ply_with_scalar_seam  # noqa: E402


DISPLAY_NAMES = {
    "threshold_curvature_best": "Curvature threshold",
    "threshold_density_best": "Density threshold",
    "threshold_linearity_best": "Linearity threshold",
    "ransac_multiplane_residual": "Multi-plane RANSAC",
}


def csv_name_from_any(name):
    if name.endswith(".ply"):
        return name[:-4] + "_label_auto.csv"
    return name


def select_cases(threshold_detail, ransac_detail):
    rows = []
    for method in (
        "threshold_curvature_best",
        "threshold_density_best",
        "threshold_linearity_best",
    ):
        group = threshold_detail[threshold_detail["method"] == method]
        rows.append((method, "good", group.loc[group["miou"].idxmax()]))
        rows.append((method, "bad", group.loc[group["miou"].idxmin()]))
    group = ransac_detail[ransac_detail["method"] == "ransac_multiplane_residual"]
    rows.append(("ransac_multiplane_residual", "good", group.loc[group["miou"].idxmax()]))
    rows.append(("ransac_multiplane_residual", "bad", group.loc[group["miou"].idxmin()]))
    return rows


def threshold_prediction(df, config):
    score = df[config["feature"]].to_numpy(dtype=np.float32)
    if config["representation"] == "within_cloud_percentile":
        score = percentile_rank(score)
    if config["direction"] == "ge":
        return score >= config["threshold"]
    return score <= config["threshold"]


def ransac_prediction(xyz_raw, config, file_index, seed, iterations, sample_cap):
    run_seed = seed + file_index
    spacing = cloud_spacing(xyz_raw, sample_cap, run_seed)
    residual, _ = fit_dominant_planes(
        xyz_raw,
        spacing,
        iterations,
        sample_cap,
        seed + file_index * 17,
        config["fit_tolerance_spacing_mult"],
    )
    return residual > config["residual_threshold_spacing_mult"]


def metric_record(y, pred):
    c = confusion(y, pred)
    return {
        "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
        **metrics(c),
    }


def stable_pca_view(xyz):
    center = np.median(xyz, axis=0)
    _, _, vh = np.linalg.svd(xyz[:: max(1, len(xyz) // 50000)] - center, full_matrices=False)
    basis = vh.copy()
    for index in range(3):
        pivot = int(np.argmax(np.abs(basis[index])))
        if basis[index, pivot] < 0:
            basis[index] *= -1
    view = (xyz - center) @ basis.T
    return view


def plot_mask(ax, xyz, mask, rng, max_background):
    mask = np.asarray(mask, dtype=bool)
    background = np.flatnonzero(~mask)
    if len(background) > max_background:
        background = rng.choice(background, max_background, replace=False)
    positive = np.flatnonzero(mask)
    ax.set_facecolor("black")
    ax.scatter(
        xyz[background, 0], xyz[background, 1], xyz[background, 2],
        c="white", s=0.12, alpha=0.58, depthshade=False, rasterized=True,
    )
    ax.scatter(
        xyz[positive, 0], xyz[positive, 1], xyz[positive, 2],
        c="#ff2020", s=0.55, alpha=1.0, depthshade=False, rasterized=True,
    )
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-62)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    limits = np.ptp(xyz, axis=0)
    ax.set_box_aspect(np.maximum(limits, np.max(limits) * 0.08))


def render_case(out_path, xyz_raw_order, y, math_pred, ours_pred, method, case, math_m, ours_m, max_background):
    view = stable_pca_view(xyz_raw_order)
    rng = np.random.default_rng(20260824)
    fig = plt.figure(figsize=(12, 4), facecolor="black")
    panels = [
        ("Ground Truth", y),
        (f"{DISPLAY_NAMES[method]}\nmIoU={math_m['miou']:.3f}, R={math_m['recall']:.3f}\nF1={math_m['f1']:.3f}, Acc={math_m['accuracy']:.3f}", math_pred),
        (f"Ours (seed 444)\nmIoU={ours_m['miou']:.3f}, R={ours_m['recall']:.3f}\nF1={ours_m['f1']:.3f}, Acc={ours_m['accuracy']:.3f}", ours_pred),
    ]
    for index, (title, mask) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        plot_mask(ax, view, mask, rng, max_background)
        ax.set_title(title, color="white", fontsize=12, pad=2)
    fig.suptitle(
        f"{DISPLAY_NAMES[method]} — {case.upper()} case",
        color="white", fontsize=14, y=0.985,
    )
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.77, wspace=0.01)
    fig.savefig(out_path, dpi=260, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="data/dataset_processed_1-17")
    parser.add_argument("--raw-root", default="data/dataset_1-17")
    parser.add_argument("--threshold-result", default="results/traditional_threshold_sweep_78812")
    parser.add_argument("--ransac-result", default="results/traditional_baselines_v2_78810")
    parser.add_argument("--ours-dir", default="results/final_benchmark_78764/predictions/ours/seed444")
    parser.add_argument("--out-dir", default="results/math_good_bad_case_study_miou")
    parser.add_argument("--max-background", type=int, default=90000)
    parser.add_argument("--ransac-iterations", type=int, default=160)
    parser.add_argument("--ransac-sample-cap", type=int, default=30000)
    parser.add_argument("--ransac-seed", type=int, default=10042)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    raw_root = Path(args.raw_root)
    threshold_root = Path(args.threshold_result)
    ransac_root = Path(args.ransac_result)
    ours_dir = Path(args.ours_dir)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "figures"
    point_dir = out_dir / "pointclouds"
    image_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    threshold_detail = pd.read_csv(threshold_root / "threshold_sweep_test_per_file.csv")
    ransac_detail = pd.read_csv(ransac_root / "test_per_file_metrics.csv")
    threshold_params = json.loads((threshold_root / "selected_thresholds.json").read_text())["selected"]
    ransac_params = json.loads((ransac_root / "selected_parameters.json").read_text())["selected_methods"]
    ransac_config = ransac_params["ransac_multiplane_residual"]
    test_files = sorted((processed_root / "test").glob("*.csv"))
    file_index = {path.name: index for index, path in enumerate(test_files)}

    selected_rows = select_cases(threshold_detail, ransac_detail)
    report_rows = []
    manifest_rows = []
    for method, case, selection_row in selected_rows:
        csv_name = csv_name_from_any(str(selection_row["file"]))
        csv_path = processed_root / "test" / csv_name
        df = pd.read_csv(csv_path)
        csv_y = df["label"].to_numpy() > 0.5
        raw_path = raw_path_for(csv_path, raw_root, "test")
        raw = read_ply_with_scalar_seam(str(raw_path))
        xyz_raw = np.asarray(raw["xyz"], dtype=np.float64)
        y_raw = np.asarray(raw["scalar_label"]) > 0.5
        permutation = np.random.default_rng(42).permutation(len(y_raw))
        if not np.array_equal(csv_y, y_raw[permutation]):
            raise ValueError(f"processed/raw alignment failed: {csv_name}")

        if method == "ransac_multiplane_residual":
            math_pred_raw = ransac_prediction(
                xyz_raw, ransac_config, file_index[csv_name], args.ransac_seed,
                args.ransac_iterations, args.ransac_sample_cap,
            )
            math_pred_csv = math_pred_raw[permutation]
        else:
            math_pred_csv = threshold_prediction(df, threshold_params[method])
            math_pred_raw = np.zeros(len(math_pred_csv), dtype=bool)
            math_pred_raw[permutation] = math_pred_csv

        ours_path = ours_dir / (csv_path.stem + "_fullpred.csv")
        ours = pd.read_csv(ours_path)
        ours_y = ours["label"].to_numpy() > 0.5
        ours_pred_csv = ours["pred"].to_numpy() > 0.5
        if not np.array_equal(csv_y, ours_y):
            raise ValueError(f"Ours/processed alignment failed: {ours_path.name}")
        ours_pred_raw = np.zeros(len(ours_pred_csv), dtype=bool)
        ours_pred_raw[permutation] = ours_pred_csv
        ours_prob_raw = np.zeros(len(ours_pred_csv), dtype=np.float32)
        ours_prob_raw[permutation] = ours["prob"].to_numpy(dtype=np.float32)

        math_m = metric_record(y_raw, math_pred_raw)
        ours_m = metric_record(y_raw, ours_pred_raw)
        for compared_method, result in ((method, math_m), ("ours_seed444", ours_m)):
            report_rows.append({
                "selection_method": method,
                "case": case,
                "file": raw_path.name,
                "evaluated_method": compared_method,
                **result,
            })

        safe_method = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
        base = f"{safe_method}_{case}_{raw_path.stem}"
        combined_path = point_dir / f"{base}.csv"
        pd.DataFrame({
            "x": xyz_raw[:, 0], "y": xyz_raw[:, 1], "z": xyz_raw[:, 2],
            "label": y_raw.astype(np.uint8),
            "math_pred": math_pred_raw.astype(np.uint8),
            "ours_pred": ours_pred_raw.astype(np.uint8),
            "ours_prob": ours_prob_raw,
        }).to_csv(combined_path, index=False)
        figure_path = image_dir / f"{base}.png"
        render_case(
            figure_path, xyz_raw, y_raw, math_pred_raw, ours_pred_raw,
            method, case, math_m, ours_m, args.max_background,
        )
        manifest_rows.append({
            "selection_method": method,
            "case": case,
            "file": raw_path.name,
            "selection_rule": "maximum test miou" if case == "good" else "minimum test miou",
            "figure": str(figure_path),
            "pointcloud_csv": str(combined_path),
        })

    metrics_df = pd.DataFrame(report_rows)
    metrics_df.to_csv(out_dir / "case_study_metrics.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "case_manifest.csv", index=False)
    requested_metrics = metrics_df[[
        "selection_method", "case", "file", "evaluated_method",
        "miou", "recall", "f1", "accuracy",
    ]].copy()
    requested_metrics.to_csv(out_dir / "case_study_requested_metrics.csv", index=False)
    pivot = requested_metrics.pivot_table(
        index=["selection_method", "case", "file"],
        columns="evaluated_method", values=["miou", "recall", "f1", "accuracy"]
    )
    pivot.to_csv(out_dir / "case_study_comparison_wide.csv")
    print(requested_metrics.to_string(index=False))
    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
