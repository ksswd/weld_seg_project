#!/usr/bin/env python3
"""Build a compact binary-PLY package for local comparison screenshots."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from render_learning_method_comparison import METHODS, match_coordinate_units, paths, read_csv, read_pointconv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/visualization/weld_all_17_types_local_visualization"


def natural_key(text: str) -> list[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", text)]


def write_binary_ply(path: Path, xyz: np.ndarray, positive: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
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


def source_stems() -> tuple[list[str], list[str]]:
    source = ROOT / "results/final_benchmark_78764/predictions/dgcnn/seed444/fullpred_test"
    stems = [p.name.removesuffix("_fullpred.csv") for p in source.glob("*_aug_18_fullpred.csv")]
    stems.sort(key=natural_key)
    primary, extras = [], []
    for number in range(1, 18):
        candidates = [s for s in stems if re.match(rf"^{number}\s*-", s)]
        preferred = next((s for s in candidates if "1Cloud" in s), candidates[0] if candidates else None)
        if preferred is None:
            raise RuntimeError(f"missing aug_18 sample for type {number}")
        primary.append(preferred)
        extras.extend(s for s in candidates if s != preferred)
    return primary, extras


def safe_name(method: str) -> str:
    return {"PointNet++": "PointNetPP"}.get(method, method)


def export_sample(stem: str, sample_dir: Path) -> None:
    sources = paths(ROOT, stem)
    reference_xyz, _ = read_csv(*sources["DGCNN"])
    for method in METHODS:
        path, target = sources[method]
        xyz, positive = read_pointconv(path) if target == "txt" else read_csv(path, target)
        xyz = match_coordinate_units(xyz, reference_xyz)
        write_binary_ply(sample_dir / f"{safe_name(method)}.ply", xyz, positive)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    primary, extras = source_stems()
    for index, stem in enumerate(primary, 1):
        print(f"[{index:02d}/17] {stem}")
        export_sample(stem, OUT / f"type_{index:02d}_{stem.replace(' ', '_')}")
    for stem in extras:
        print(f"[extra] {stem}")
        export_sample(stem, OUT / "additional_variants" / stem.replace(" ", "_"))

    readme = """# Weld segmentation local visualization package

Each sample directory contains the same physical point cloud for:
GT, DGCNN, PTv3, PointNetPP (fixed), PointMLP, PointConv, and Ours.

- Binary little-endian PLY; compatible with CloudCompare and MeshLab.
- White points: background.
- Red points: ground-truth/predicted weld.
- Set the viewer canvas/background to black.
- The 17 primary directories use aug_18. Types 1 and 2 have a second geometry,
  preserved under additional_variants.
- All methods are mapped to the same raw-coordinate frame so a saved camera can
  be reused across methods in the same sample directory.
- PTv3 genuinely degenerates on Type 17; the red panel is not a rendering bug.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved package directory: {OUT}")


if __name__ == "__main__":
    main()
