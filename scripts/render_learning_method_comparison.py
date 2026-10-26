#!/usr/bin/env python3
"""Render the final weld baseline predictions with one camera per sample row."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("GT", "DGCNN", "PointNet++", "PointMLP", "PointConv", "Ours")
SAMPLES = (
    ("Type 1", "1 - 1Cloud_aug_18"),
    ("Type 13", "13 - Cloud_aug_18"),
    ("Type 17", "17 - Cloud_aug_18"),
)


def read_csv(path: Path, target: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=["x", "y", "z", target])
    return df[["x", "y", "z"]].to_numpy(np.float64), (df[target].to_numpy() > 0.5)


def read_pointconv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :3], arr[:, 4] > 0.5


def paths(root: Path, stem: str) -> dict[str, tuple[Path, str]]:
    old = root / "results/final_benchmark_78764/predictions"
    return {
        "GT": (old / "ours/seed444" / f"{stem}_label_auto_fullpred.csv", "label"),
        "DGCNN": (old / "dgcnn/seed444/fullpred_test" / f"{stem}_fullpred.csv", "pred"),
        "PointNet++": (
            root / "results/pointnet2_fixed_78773/predictions/seed444/fullpred_test" / f"{stem}_fullpred.csv",
            "pred",
        ),
        "PointMLP": (old / "pointmlp/seed444/fullpred_test" / f"{stem}_label_auto.csv", "pred"),
        "PointConv": (old / "pointconv/seed444/pred_xyz_label_test" / f"{stem}.txt", "txt"),
        "Ours": (old / "ours/seed444" / f"{stem}_label_auto_fullpred.csv", "pred"),
    }


def pca_camera(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(xyz, axis=0)
    centered = xyz - center
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    camera = vectors[:, np.argsort(values)[::-1][:2]]
    for j in range(2):
        dominant = int(np.argmax(np.abs(camera[:, j])))
        if camera[dominant, j] < 0:
            camera[:, j] *= -1
    return center, camera


def project(xyz: np.ndarray, center: np.ndarray, camera: np.ndarray) -> np.ndarray:
    return (xyz - center) @ camera


def match_coordinate_units(xyz: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map per-axis preprocessed coordinates back to the raw-coordinate frame."""
    q = (0.001, 0.999)
    src_lo, src_hi = np.quantile(xyz, q, axis=0)
    ref_lo, ref_hi = np.quantile(reference, q, axis=0)
    scale = (ref_hi - ref_lo) / np.maximum(src_hi - src_lo, 1e-12)
    return (xyz - src_lo) * scale + ref_lo


def limits(uv: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lo = np.quantile(uv, 0.001, axis=0)
    hi = np.quantile(uv, 0.999, axis=0)
    margin = np.maximum((hi - lo) * 0.025, 1e-6)
    return (lo[0] - margin[0], hi[0] + margin[0]), (lo[1] - margin[1], hi[1] + margin[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path,
                        default=Path("results/visualization/final_learning_methods_comparison_1_9_17.png"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SAMPLES), len(METHODS), figsize=(18, 9), facecolor="black")
    for row, (row_name, stem) in enumerate(SAMPLES):
        sources = paths(root, stem)
        # DGCNN/PointNet++/PointConv retain raw millimetre coordinates.  Use
        # that common frame as the row camera reference.
        reference_xyz, _ = read_csv(*sources["DGCNN"])
        center, camera = pca_camera(reference_xyz)
        reference_uv = project(reference_xyz, center, camera)
        xlim, ylim = limits(reference_uv)
        for col, method in enumerate(METHODS):
            path, target = sources[method]
            if not path.exists():
                raise FileNotFoundError(path)
            xyz, positive = read_pointconv(path) if target == "txt" else read_csv(path, target)
            # Ours/PointMLP use a per-axis preprocessed coordinate frame. Map
            # all copies to the same raw frame before applying one shared
            # camera, without changing point labels or predictions.
            xyz = match_coordinate_units(xyz, reference_xyz)
            uv = project(xyz, center, camera)
            ax = axes[row, col]
            ax.set_facecolor("black")
            # Draw negatives first and positives last so weld points remain visible.
            negative = ~positive
            ax.scatter(uv[negative, 0], uv[negative, 1], s=0.09, c="#FFFFFF",
                       alpha=0.78, linewidths=0, rasterized=True)
            ax.scatter(uv[positive, 0], uv[positive, 1], s=0.34, c="#FF2020",
                       alpha=1.0, linewidths=0, rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(method, color="white", fontsize=15, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(row_name, color="white", fontsize=14, fontweight="bold", labelpad=8)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.95, bottom=0.015, wspace=0.018, hspace=0.045)
    fig.savefig(output, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved PNG: {output}")
    print(f"Saved PDF: {pdf}")


if __name__ == "__main__":
    main()
