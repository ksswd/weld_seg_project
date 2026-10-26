#!/usr/bin/env python3
"""Export mathematical good/bad case-study CSVs to baseline-compatible PLY."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_NAMES = {
    "threshold_curvature_best": "CurvatureThreshold",
    "threshold_density_best": "DensityThreshold",
    "threshold_linearity_best": "LinearityThreshold",
    "ransac_multiplane_residual": "MultiPlaneRANSAC",
}


def write_binary_ply(path: Path, xyz: np.ndarray, positive: np.ndarray) -> None:
    """Match the previous baseline package: raw XYZ, white background, red weld."""
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = np.asarray(positive, dtype=bool)
    vertex = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    xyz = np.asarray(xyz, dtype=np.float32)
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"] = 255
    vertex["green"] = np.where(positive, 32, 255).astype(np.uint8)
    vertex["blue"] = np.where(positive, 32, 255).astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex.tofile(handle)


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", default="results/math_good_bad_case_study_miou")
    parser.add_argument("--out-dir", default="results/math_good_bad_case_study_miou_ply")
    args = parser.parse_args()
    case_root = Path(args.case_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(case_root / "case_manifest.csv")
    for row in manifest.itertuples(index=False):
        method = str(row.selection_method)
        case = str(row.case)
        display = METHOD_NAMES[method]
        csv_path = Path(row.pointcloud_csv)
        if not csv_path.is_absolute():
            # Manifest paths are repository-relative in the current experiment.
            csv_path = Path.cwd() / csv_path
        df = pd.read_csv(csv_path)
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        folder = out_dir / f"{display}_{case}_{safe_stem(Path(str(row.file)).stem)}"
        write_binary_ply(folder / "GT.ply", xyz, df["label"].to_numpy() > 0.5)
        write_binary_ply(folder / f"{display}.ply", xyz, df["math_pred"].to_numpy() > 0.5)
        write_binary_ply(folder / "Ours.ply", xyz, df["ours_pred"].to_numpy() > 0.5)
        print(f"[PLY] {folder}")

    readme = """# Mathematical good/bad cases — PLY package

Each folder contains the identical raw-coordinate point cloud rendered as:

- `GT.ply`: ground-truth weld
- the named mathematical method PLY: mathematical prediction
- `Ours.ply`: Ours seed-444 prediction

Format matches the previous learning-baseline visualization package:

- binary little-endian PLY
- x/y/z float32 plus RGB uchar
- white points: background
- red points: weld
- set the viewer background to black

Good and bad cases were selected by maximum/minimum per-file test mIoU of the
corresponding mathematical method. Ours did not participate in sample selection.
All three PLY files in one folder have identical point order and coordinates, so
the same saved camera/view can be applied directly.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest.to_csv(out_dir / "case_manifest.csv", index=False)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
