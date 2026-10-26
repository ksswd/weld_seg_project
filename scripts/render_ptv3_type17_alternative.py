#!/usr/bin/env python3
"""Render the least-degenerate archived PTv3 Type-17 prediction."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_learning_method_comparison import limits, pca_camera, project


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "comparable_methods/Pointcept-copy/data/weld/test/17 - Cloud_aug_18"
PRED = ROOT / "comparable_methods/Pointcept-copy/exp/weld/weld_newdata_200epochs/result/17 - Cloud_aug_18_pred.npy"
OUT = ROOT / "results/visualization/ptv3_type17_alternative_gt_vs_pred.png"


def main() -> None:
    xyz = np.load(DATA / "coord.npy").astype(np.float64)
    label = np.load(DATA / "segment.npy").reshape(-1) > 0
    pred = np.load(PRED).reshape(-1) > 0
    center, camera = pca_camera(xyz)
    uv = project(xyz, center, camera)
    xlim, ylim = limits(uv)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor="black")
    for ax, positive, title in zip(axes, (label, pred), ("GT", "PTv3 (archived run)")):
        ax.set_facecolor("black")
        ax.scatter(uv[~positive, 0], uv[~positive, 1], s=0.16, c="white",
                   alpha=0.78, linewidths=0, rasterized=True)
        ax.scatter(uv[positive, 0], uv[positive, 1], s=0.55, c="#FF2020",
                   linewidths=0, rasterized=True)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, color="white", weight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01, wspace=0.04)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="black", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="black", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
