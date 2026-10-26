#!/usr/bin/env python3
"""Compute weld centerline localization errors from full point predictions.

Protocol
--------
1. Extract GT and predicted weld-mask points independently.
2. Apply the same PCA-ordered cross-sectional-centroid extractor to both masks.
3. Resample both open curves at equal normalized arc-length positions.
4. Report the smaller forward/reversed point-correspondence RMSE.

All distances are computed in the original PLY coordinate system (millimetres for
the current weld dataset).  The script understands the output layouts produced
by the final unified benchmark, the fixed PointNet++ run, and the recovered PTv3
predictions. An optional kNN-graph/geodesic extractor is retained for sensitivity
checks, but is not the default because several corner-weld masks contain two
nearby surface strips that must jointly define one physical centre trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: str
    prediction_dir: Path


@dataclass
class CenterlineResult:
    curve: Optional[np.ndarray]
    status: str
    input_points: int
    graph_points: int
    component_points: int
    component_fraction: float
    length: float


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def read_ascii_ply_xyz_label(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read xyz and scalar_label from an ASCII PLY without optional packages."""
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = None
        properties: List[str] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == "format" and fields[1] != "ascii":
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if fields[0] == "element":
                in_vertex = fields[1] == "vertex"
                if in_vertex:
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and in_vertex:
                properties.append(fields[-1])
            elif fields[0] == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        data = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] != vertex_count:
        raise ValueError(
            f"PLY vertex count mismatch in {path}: {data.shape[0]} != {vertex_count}"
        )
    col = {name: idx for idx, name in enumerate(properties)}
    for name in ("x", "y", "z"):
        if name not in col:
            raise ValueError(f"PLY is missing {name}: {path}")
    label_name = next(
        (name for name in ("scalar_label", "label", "segment") if name in col), None
    )
    if label_name is None:
        raise ValueError(f"PLY has no scalar_label/label property: {path}")
    xyz = data[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    label = (data[:, col[label_name]] > 0.5).astype(np.uint8)
    return xyz, label


def canonical_stem(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_label_auto_fullpred",
        "_label_auto_pred",
        "_fullpred",
        "_label_auto",
        "_pred",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def prediction_files(directory: Path) -> List[Path]:
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".txt"}
    ]
    excluded_prefixes = (
        "inference_", "eval_", "summary_", "benchmark_", "experiment_",
    )
    return sorted(
        (path for path in files if not path.stem.startswith(excluded_prefixes)),
        key=lambda path: canonical_stem(path),
    )


def read_prediction(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return xyz, predicted binary mask, and optional saved GT mask."""
    if path.suffix.lower() == ".txt":
        data = np.loadtxt(path, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 4:
            raise ValueError(f"Expected at least four columns in {path}")
        xyz = data[:, :3]
        if data.shape[1] >= 5:
            saved_gt = (data[:, 3] > 0.5).astype(np.uint8)
            pred = (data[:, 4] > 0.5).astype(np.uint8)
        else:
            saved_gt = None
            pred = (data[:, 3] > 0.5).astype(np.uint8)
        return xyz, pred, saved_gt

    frame = pd.read_csv(path)
    required_xyz = ["x", "y", "z"]
    if not set(required_xyz).issubset(frame.columns):
        raise ValueError(f"CSV lacks x/y/z columns: {path}")
    xyz = frame[required_xyz].to_numpy(dtype=np.float64)
    if "pred" in frame.columns:
        pred = (frame["pred"].to_numpy(dtype=np.float64) > 0.5).astype(np.uint8)
    elif "prediction" in frame.columns:
        pred = (frame["prediction"].to_numpy(dtype=np.float64) > 0.5).astype(np.uint8)
    else:
        raise ValueError(f"CSV lacks pred/prediction column: {path}")
    gt_col = next((name for name in ("label", "gt", "target") if name in frame.columns), None)
    saved_gt = None
    if gt_col is not None:
        saved_gt = (frame[gt_col].to_numpy(dtype=np.float64) > 0.5).astype(np.uint8)
    return xyz, pred, saved_gt


def sample_coordinate_match(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    if a.shape != b.shape:
        return False
    if len(a) == 0:
        return True
    idx = np.linspace(0, len(a) - 1, min(128, len(a)), dtype=np.int64)
    return bool(np.allclose(a[idx], b[idx], rtol=0.0, atol=atol))


def resolve_raw_order(
    prediction_xyz: np.ndarray,
    raw_xyz: np.ndarray,
    processed_csv: Optional[Path],
    shuffle_seed: int,
    coordinate_tolerance: float,
) -> Tuple[np.ndarray, str]:
    """Return raw coordinates in the same row order as a prediction file."""
    if prediction_xyz.shape[0] != raw_xyz.shape[0]:
        raise ValueError(
            f"Prediction/raw point-count mismatch: {prediction_xyz.shape[0]} vs {raw_xyz.shape[0]}"
        )
    if sample_coordinate_match(prediction_xyz, raw_xyz, coordinate_tolerance):
        return raw_xyz, "raw"

    if processed_csv is not None and processed_csv.exists():
        processed_xyz = pd.read_csv(
            processed_csv, usecols=["x", "y", "z"]
        ).to_numpy(dtype=np.float64)
        if sample_coordinate_match(prediction_xyz, processed_xyz, coordinate_tolerance):
            permutation = np.random.default_rng(shuffle_seed).permutation(len(raw_xyz))
            return raw_xyz[permutation], "processed_shuffle"

    # A final raw-coordinate fallback handles harmless row reordering. It is only
    # attempted when values are in the same physical scale.
    pred_extent = np.linalg.norm(np.ptp(prediction_xyz, axis=0))
    raw_extent = np.linalg.norm(np.ptp(raw_xyz, axis=0))
    scale_ratio = pred_extent / max(raw_extent, 1e-12)
    if 0.5 <= scale_ratio <= 2.0:
        tree = cKDTree(raw_xyz)
        distances, indices = tree.query(prediction_xyz, k=1, workers=-1)
        if np.max(distances) <= coordinate_tolerance and len(np.unique(indices)) == len(indices):
            return raw_xyz[indices], "raw_kdtree"

    raise ValueError(
        "Could not map prediction rows to original coordinates; neither raw nor "
        "processed coordinate order matched"
    )


def resample_polyline(points: np.ndarray, count: int) -> Tuple[np.ndarray, float]:
    if len(points) < 2:
        raise ValueError("A polyline needs at least two points")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, segment > 1e-12]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Polyline has zero length")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment)]
    total = float(arc[-1])
    target = np.linspace(0.0, total, count)
    curve = np.column_stack([np.interp(target, arc, points[:, dim]) for dim in range(3)])
    return curve, total


def extract_centerline(
    points: np.ndarray,
    *,
    output_points: int,
    station_count: int,
    knn: int,
    max_graph_points: int,
    min_mask_points: int,
    edge_factor: float,
    smoothing_window: int,
    random_seed: int,
    refinement_iterations: int = 0,
) -> CenterlineResult:
    input_count = int(len(points))
    if input_count < min_mask_points:
        return CenterlineResult(None, "too_few_mask_points", input_count, 0, 0, 0.0, math.nan)

    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < min_mask_points:
        return CenterlineResult(None, "nonfinite_or_too_few", input_count, 0, 0, 0.0, math.nan)

    # Exact duplicates create zero-weight graph edges and unstable components.
    points = np.unique(points, axis=0)
    if len(points) > max_graph_points:
        rng = np.random.default_rng(random_seed)
        chosen = np.sort(rng.choice(len(points), size=max_graph_points, replace=False))
        graph_points = points[chosen]
    else:
        graph_points = points
    graph_count = int(len(graph_points))
    if graph_count < min_mask_points:
        return CenterlineResult(None, "too_few_unique_points", input_count, graph_count, 0, 0.0, math.nan)

    k = min(max(2, knn), graph_count - 1)
    tree = cKDTree(graph_points)
    distances, neighbours = tree.query(graph_points, k=k + 1, workers=-1)
    nearest = distances[:, 1]
    kth = distances[:, -1]
    positive_nn = nearest[nearest > 0]
    if len(positive_nn) == 0:
        return CenterlineResult(None, "degenerate_neighbours", input_count, graph_count, 0, 0.0, math.nan)
    # Robustly reject long edges that would join separate false-positive islands.
    cutoff = max(
        float(np.percentile(positive_nn, 95)) * edge_factor,
        float(np.percentile(kth[np.isfinite(kth)], 90)) * 1.25,
    )

    rows = np.repeat(np.arange(graph_count), k)
    cols = neighbours[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)
    valid = np.isfinite(weights) & (weights > 0) & (weights <= cutoff)
    graph = coo_matrix(
        (weights[valid], (rows[valid], cols[valid])),
        shape=(graph_count, graph_count),
    ).tocsr()
    graph = graph.maximum(graph.T)
    component_count, labels = connected_components(graph, directed=False)
    if component_count == 0:
        return CenterlineResult(None, "no_graph_component", input_count, graph_count, 0, 0.0, math.nan)
    sizes = np.bincount(labels)
    component_label = int(np.argmax(sizes))
    component_idx = np.flatnonzero(labels == component_label)
    component_size = int(len(component_idx))
    component_fraction = component_size / graph_count
    if component_size < min_mask_points:
        return CenterlineResult(
            None, "largest_component_too_small", input_count, graph_count,
            component_size, component_fraction, math.nan
        )

    component_graph = graph[component_idx][:, component_idx]
    component_xyz = graph_points[component_idx]
    first_dist = dijkstra(component_graph, directed=False, indices=0)
    endpoint_a = int(np.argmax(first_dist))
    geodesic = dijkstra(component_graph, directed=False, indices=endpoint_a)
    finite = np.isfinite(geodesic)
    if np.count_nonzero(finite) < min_mask_points:
        return CenterlineResult(
            None, "invalid_geodesic", input_count, graph_count,
            component_size, component_fraction, math.nan
        )
    component_xyz = component_xyz[finite]
    geodesic = geodesic[finite]
    diameter = float(np.max(geodesic))
    if not np.isfinite(diameter) or diameter <= 1e-12:
        return CenterlineResult(
            None, "zero_geodesic_length", input_count, graph_count,
            component_size, component_fraction, math.nan
        )

    bins = np.linspace(0.0, diameter, station_count + 1)
    bin_id = np.clip(np.digitize(geodesic, bins[1:-1]), 0, station_count - 1)
    stations = []
    for idx in range(station_count):
        values = component_xyz[bin_id == idx]
        if len(values):
            stations.append(np.median(values, axis=0))
    if len(stations) < 4:
        return CenterlineResult(
            None, "too_few_center_stations", input_count, graph_count,
            component_size, component_fraction, math.nan
        )
    stations_np = np.asarray(stations, dtype=np.float64)

    window = min(smoothing_window, len(stations_np) if len(stations_np) % 2 else len(stations_np) - 1)
    if window >= 5:
        for dim in range(3):
            stations_np[:, dim] = savgol_filter(
                stations_np[:, dim], window_length=window, polyorder=2, mode="interp"
            )
    try:
        curve, length = resample_polyline(stations_np, output_points)
    except ValueError:
        return CenterlineResult(
            None, "polyline_resampling_failed", input_count, graph_count,
            component_size, component_fraction, math.nan
        )
    return CenterlineResult(
        curve, "ok", input_count, graph_count, component_size, component_fraction, length
    )


def extract_centerline_principal_curve(
    points: np.ndarray,
    *,
    output_points: int,
    station_count: int,
    max_graph_points: int,
    min_mask_points: int,
    smoothing_window: int,
    random_seed: int,
    refinement_iterations: int,
    **_: object,
) -> CenterlineResult:
    """Extract a robust 1-D centre trajectory from a possibly multi-ridge mask.

    Weld labels on corner joints can form two close, disconnected strips on the
    adjoining surfaces. Selecting only the largest connected component would
    choose one side instead of the physical centre. This extractor therefore
    initializes an ordering with PCA, takes cross-sectional centroids
    over *all* mask points, and iteratively refines assignments to the current
    principal curve. The same deterministic operation is applied to GT and pred.
    """
    input_count = int(len(points))
    if input_count < min_mask_points:
        return CenterlineResult(None, "too_few_mask_points", input_count, 0, 0, 0.0, math.nan)
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = np.unique(points, axis=0)
    if len(points) < min_mask_points:
        return CenterlineResult(None, "nonfinite_or_too_few", input_count, 0, 0, 0.0, math.nan)
    # PCA/binning is linear in the number of mask points, so retain every point.
    # This avoids introducing method-dependent Monte-Carlo variation. The
    # max_graph_points limit is used only by the optional graph extractor.
    work = points
    work_count = int(len(work))

    centred = work - np.mean(work, axis=0, keepdims=True)
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    parameter = centred @ axis

    def median_stations(values: np.ndarray) -> Optional[np.ndarray]:
        lo, hi = float(np.min(values)), float(np.max(values))
        if not np.isfinite(lo + hi) or hi - lo <= 1e-12:
            return None
        edges = np.linspace(lo, hi, station_count + 1)
        ids = np.clip(np.digitize(values, edges[1:-1]), 0, station_count - 1)
        result = [np.mean(work[ids == idx], axis=0) for idx in range(station_count) if np.any(ids == idx)]
        if len(result) < 4:
            return None
        return np.asarray(result, dtype=np.float64)

    stations = median_stations(parameter)
    if stations is None:
        return CenterlineResult(None, "pca_station_failure", input_count, work_count, work_count, 1.0, math.nan)

    for _iteration in range(max(0, refinement_iterations)):
        try:
            guide, _ = resample_polyline(stations, station_count)
        except ValueError:
            break
        nearest_station = cKDTree(guide).query(work, k=1, workers=-1)[1]
        refined = []
        for idx in range(station_count):
            values = work[nearest_station == idx]
            if len(values):
                refined.append((idx, np.mean(values, axis=0)))
        if len(refined) < 4:
            break
        station_ids = np.asarray([item[0] for item in refined], dtype=np.float64)
        station_xyz = np.asarray([item[1] for item in refined], dtype=np.float64)
        complete_ids = np.arange(station_count, dtype=np.float64)
        stations = np.column_stack(
            [np.interp(complete_ids, station_ids, station_xyz[:, dim]) for dim in range(3)]
        )

    window = min(smoothing_window, len(stations) if len(stations) % 2 else len(stations) - 1)
    if window >= 5:
        for dim in range(3):
            stations[:, dim] = savgol_filter(
                stations[:, dim], window_length=window, polyorder=2, mode="interp"
            )
    try:
        curve, length = resample_polyline(stations, output_points)
    except ValueError:
        return CenterlineResult(None, "polyline_resampling_failed", input_count, work_count, work_count, 1.0, math.nan)
    return CenterlineResult(curve, "ok", input_count, work_count, work_count, 1.0, length)


def curve_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    forward = float(np.sqrt(np.mean(np.sum((gt - pred) ** 2, axis=1))))
    reverse = float(np.sqrt(np.mean(np.sum((gt - pred[::-1]) ** 2, axis=1))))
    reversed_order = reverse < forward
    aligned = pred[::-1] if reversed_order else pred
    rmse = min(forward, reverse)
    correspondence_errors = np.linalg.norm(gt - aligned, axis=1)

    pred_tree = cKDTree(aligned)
    gt_tree = cKDTree(gt)
    d_gt_pred = pred_tree.query(gt, k=1, workers=-1)[0]
    d_pred_gt = gt_tree.query(aligned, k=1, workers=-1)[0]
    chamfer_rmse = float(
        np.sqrt(0.5 * (np.mean(d_gt_pred**2) + np.mean(d_pred_gt**2)))
    )
    endpoint_rmse = float(
        np.sqrt(0.5 * (np.sum((gt[0] - aligned[0]) ** 2) + np.sum((gt[-1] - aligned[-1]) ** 2)))
    )
    return {
        "rmse": rmse,
        "rmse_forward": forward,
        "rmse_reverse": reverse,
        "reversed": int(reversed_order),
        "chamfer_rmse": chamfer_rmse,
        "endpoint_rmse": endpoint_rmse,
        "max_error": float(np.max(correspondence_errors)),
    }


def discover_default_runs(root: Path) -> List[RunSpec]:
    final_root = root / "results/final_benchmark_78764/predictions"
    specs: List[RunSpec] = []
    layouts = {
        "ours": (final_root / "ours", "."),
        "pointmlp": (final_root / "pointmlp", "fullpred_test"),
        "dgcnn": (final_root / "dgcnn", "fullpred_test"),
        "pointconv": (final_root / "pointconv", "pred_xyz_label_test"),
    }
    for method, (method_root, child) in layouts.items():
        if not method_root.exists():
            continue
        for seed_dir in sorted(method_root.glob("seed*")):
            pred_dir = seed_dir if child == "." else seed_dir / child
            if pred_dir.is_dir() and prediction_files(pred_dir):
                specs.append(RunSpec(method, seed_dir.name.removeprefix("seed"), pred_dir))

    pn_root = root / "results/pointnet2_fixed_78773/predictions"
    if pn_root.exists():
        for seed_dir in sorted(pn_root.glob("seed*")):
            pred_dir = seed_dir / "fullpred_test"
            if pred_dir.is_dir() and prediction_files(pred_dir):
                specs.append(RunSpec("pointnet2", seed_dir.name.removeprefix("seed"), pred_dir))

    ptv3_dir = root / "results/visualization/ptv3_fullpred"
    if ptv3_dir.is_dir() and prediction_files(ptv3_dir):
        specs.append(RunSpec("ptv3", "single", ptv3_dir))
    return specs


def parse_custom_runs(items: Sequence[str]) -> List[RunSpec]:
    specs = []
    for item in items:
        parts = item.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid --run {item!r}; expected METHOD[:SEED]=DIRECTORY")
        label, directory = parts
        if ":" in label:
            method, seed = label.split(":", 1)
        else:
            method, seed = label, "single"
        specs.append(RunSpec(method, seed, Path(directory).resolve()))
    return specs


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nan_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-test-dir", type=Path, default=Path("data/dataset_1-17/test"))
    parser.add_argument("--processed-test-dir", type=Path, default=Path("data/dataset_processed_1-17/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/centerline_rmse"))
    parser.add_argument(
        "--run", action="append", default=[], metavar="METHOD[:SEED]=DIR",
        help="Custom prediction directory; repeat for multiple runs. Defaults discover final benchmark outputs.",
    )
    parser.add_argument("--only-methods", nargs="*", default=[])
    parser.add_argument("--resample-points", type=int, default=100)
    parser.add_argument("--stations", type=int, default=64)
    parser.add_argument(
        "--extractor", choices=("principal_curve", "graph"), default="principal_curve",
        help="principal_curve handles the two close strips present in some corner-weld masks",
    )
    parser.add_argument("--refinement-iterations", type=int, default=0)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--max-graph-points", type=int, default=8000)
    parser.add_argument("--min-mask-points", type=int, default=30)
    parser.add_argument("--edge-factor", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--coordinate-tolerance", type=float, default=2e-3)
    parser.add_argument("--limit-files", type=int, default=0, help="For smoke tests; 0 means all files")
    args = parser.parse_args()

    root = args.project_root.resolve()
    raw_dir = args.raw_test_dir if args.raw_test_dir.is_absolute() else root / args.raw_test_dir
    processed_dir = (
        args.processed_test_dir if args.processed_test_dir.is_absolute()
        else root / args.processed_test_dir
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_custom_runs(args.run) if args.run else discover_default_runs(root)
    if args.only_methods:
        allowed = set(args.only_methods)
        runs = [run for run in runs if run.method in allowed]
    if not runs:
        parser.error("No prediction runs were found. Supply one or more --run arguments.")

    print(f"[CONFIG] raw={raw_dir}")
    print(f"[CONFIG] processed={processed_dir}")
    print(f"[CONFIG] output={output_dir}")
    print("[PROTOCOL] same extractor for GT/pred; normalized arc length; forward/reverse minimum")
    print(f"[EXTRACTOR] {args.extractor}")
    print("[RUNS] " + " ".join(f"{r.method}:{r.seed}" for r in runs))

    gt_cache: Dict[str, Tuple[np.ndarray, np.ndarray, CenterlineResult]] = {}
    rows: List[Dict[str, object]] = []

    for run in runs:
        files = prediction_files(run.prediction_dir)
        if args.limit_files > 0:
            files = files[: args.limit_files]
        print(f"\n[RUN] method={run.method} seed={run.seed} files={len(files)} dir={run.prediction_dir}")
        for file_index, pred_path in enumerate(files, start=1):
            stem = canonical_stem(pred_path)
            raw_path = raw_dir / f"{stem}.ply"
            processed_path = processed_dir / f"{stem}_label_auto.csv"
            base: Dict[str, object] = {
                "method": run.method,
                "seed": run.seed,
                "file": stem,
                "status": "error",
                "coordinate_order": "",
                "gt_mask_points": 0,
                "pred_mask_points": 0,
                "gt_component_fraction": math.nan,
                "pred_component_fraction": math.nan,
                "gt_length_mm": math.nan,
                "pred_length_mm": math.nan,
                "rmse_mm": math.nan,
                "chamfer_rmse_mm": math.nan,
                "endpoint_rmse_mm": math.nan,
                "max_error_mm": math.nan,
                "reversed": "",
                "message": "",
            }
            try:
                if not raw_path.exists():
                    raise FileNotFoundError(f"Missing raw PLY: {raw_path}")
                prediction_xyz, pred_mask, saved_gt = read_prediction(pred_path)
                if stem not in gt_cache:
                    raw_xyz, raw_gt = read_ascii_ply_xyz_label(raw_path)
                    extractor = (
                        extract_centerline_principal_curve
                        if args.extractor == "principal_curve" else extract_centerline
                    )
                    gt_result = extractor(
                        raw_xyz[raw_gt > 0],
                        output_points=args.resample_points,
                        station_count=args.stations,
                        knn=args.knn,
                        max_graph_points=args.max_graph_points,
                        min_mask_points=args.min_mask_points,
                        edge_factor=args.edge_factor,
                        smoothing_window=args.smoothing_window,
                        random_seed=stable_seed(f"gt:{stem}"),
                        refinement_iterations=args.refinement_iterations,
                    )
                    gt_cache[stem] = (raw_xyz, raw_gt, gt_result)
                raw_xyz, raw_gt, gt_result = gt_cache[stem]
                ordered_xyz, coordinate_order = resolve_raw_order(
                    prediction_xyz,
                    raw_xyz,
                    processed_path if processed_path.exists() else None,
                    args.shuffle_seed,
                    args.coordinate_tolerance,
                )
                if saved_gt is not None:
                    ordered_raw_gt = raw_gt
                    if coordinate_order == "processed_shuffle":
                        permutation = np.random.default_rng(args.shuffle_seed).permutation(len(raw_gt))
                        ordered_raw_gt = raw_gt[permutation]
                    elif coordinate_order == "raw_kdtree":
                        tree = cKDTree(raw_xyz)
                        _, mapping = tree.query(prediction_xyz, k=1, workers=-1)
                        ordered_raw_gt = raw_gt[mapping]
                    mismatch = float(np.mean(saved_gt != ordered_raw_gt))
                    if mismatch > 1e-6:
                        raise ValueError(f"Saved GT disagrees with raw PLY for {mismatch:.3%} of points")

                extractor = (
                    extract_centerline_principal_curve
                    if args.extractor == "principal_curve" else extract_centerline
                )
                pred_result = extractor(
                    ordered_xyz[pred_mask > 0],
                    output_points=args.resample_points,
                    station_count=args.stations,
                    knn=args.knn,
                    max_graph_points=args.max_graph_points,
                    min_mask_points=args.min_mask_points,
                    edge_factor=args.edge_factor,
                    smoothing_window=args.smoothing_window,
                    random_seed=stable_seed(f"pred:{run.method}:{run.seed}:{stem}"),
                    refinement_iterations=args.refinement_iterations,
                )
                base.update(
                    {
                        "coordinate_order": coordinate_order,
                        "gt_mask_points": int(np.count_nonzero(raw_gt)),
                        "pred_mask_points": int(np.count_nonzero(pred_mask)),
                        "gt_component_fraction": gt_result.component_fraction,
                        "pred_component_fraction": pred_result.component_fraction,
                        "gt_length_mm": gt_result.length,
                        "pred_length_mm": pred_result.length,
                    }
                )
                if gt_result.curve is None:
                    base["status"] = "gt_extraction_failed"
                    base["message"] = gt_result.status
                elif pred_result.curve is None:
                    base["status"] = "pred_extraction_failed"
                    base["message"] = pred_result.status
                else:
                    metrics = curve_metrics(gt_result.curve, pred_result.curve)
                    base.update(
                        {
                            "status": "ok",
                            "rmse_mm": metrics["rmse"],
                            "chamfer_rmse_mm": metrics["chamfer_rmse"],
                            "endpoint_rmse_mm": metrics["endpoint_rmse"],
                            "max_error_mm": metrics["max_error"],
                            "reversed": metrics["reversed"],
                            "message": "",
                        }
                    )
            except Exception as exc:  # keep all failures visible in the result table
                base["status"] = "error"
                base["message"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            if file_index == 1 or file_index % 10 == 0 or file_index == len(files):
                print(f"  [{file_index:>2}/{len(files)}] {stem}: {base['status']}")

    per_file_fields = [
        "method", "seed", "file", "status", "coordinate_order",
        "gt_mask_points", "pred_mask_points", "gt_component_fraction",
        "pred_component_fraction", "gt_length_mm", "pred_length_mm",
        "rmse_mm", "chamfer_rmse_mm", "endpoint_rmse_mm", "max_error_mm",
        "reversed", "message",
    ]
    write_csv(output_dir / "centerline_metrics_per_file.csv", rows, per_file_fields)

    run_rows: List[Dict[str, object]] = []
    for run in runs:
        subset = [r for r in rows if r["method"] == run.method and r["seed"] == run.seed]
        valid = [r for r in subset if r["status"] == "ok"]
        rmse_stats = nan_stats([float(r["rmse_mm"]) for r in valid])
        chamfer_stats = nan_stats([float(r["chamfer_rmse_mm"]) for r in valid])
        run_rows.append(
            {
                "method": run.method,
                "seed": run.seed,
                "num_files": len(subset),
                "valid_files": len(valid),
                "valid_rate": len(valid) / len(subset) if subset else math.nan,
                "rmse_mean_mm": rmse_stats["mean"],
                "rmse_std_mm": rmse_stats["std"],
                "rmse_median_mm": rmse_stats["median"],
                "rmse_p95_mm": rmse_stats["p95"],
                "chamfer_rmse_mean_mm": chamfer_stats["mean"],
                "chamfer_rmse_median_mm": chamfer_stats["median"],
                "max_error_mm": max(
                    (float(r["max_error_mm"]) for r in valid), default=math.nan
                ),
            }
        )
    run_fields = [
        "method", "seed", "num_files", "valid_files", "valid_rate",
        "rmse_mean_mm", "rmse_std_mm", "rmse_median_mm", "rmse_p95_mm",
        "chamfer_rmse_mean_mm", "chamfer_rmse_median_mm", "max_error_mm",
    ]
    write_csv(output_dir / "centerline_metrics_per_run.csv", run_rows, run_fields)

    method_rows: List[Dict[str, object]] = []
    for method in sorted({r.method for r in runs}):
        method_run_rows = [r for r in run_rows if r["method"] == method]
        run_rmse = [float(r["rmse_mean_mm"]) for r in method_run_rows]
        run_chamfer = [float(r["chamfer_rmse_mean_mm"]) for r in method_run_rows]
        run_max_error = [float(r["max_error_mm"]) for r in method_run_rows]
        rmse_stats = nan_stats(run_rmse)
        chamfer_stats = nan_stats(run_chamfer)
        max_error_stats = nan_stats(run_max_error)
        total_files = sum(int(r["num_files"]) for r in method_run_rows)
        valid_files = sum(int(r["valid_files"]) for r in method_run_rows)
        method_rows.append(
            {
                "method": method,
                "runs": len(method_run_rows),
                "total_files": total_files,
                "valid_files": valid_files,
                "valid_rate": valid_files / total_files if total_files else math.nan,
                "rmse_mean_mm": rmse_stats["mean"],
                "rmse_between_run_std_mm": rmse_stats["std"],
                "chamfer_rmse_mean_mm": chamfer_stats["mean"],
                "chamfer_between_run_std_mm": chamfer_stats["std"],
                "max_error_mean_across_runs_mm": max_error_stats["mean"],
                "max_error_between_run_std_mm": max_error_stats["std"],
                "max_error_worst_run_mm": max(run_max_error) if run_max_error else math.nan,
            }
        )
    method_fields = [
        "method", "runs", "total_files", "valid_files", "valid_rate",
        "rmse_mean_mm", "rmse_between_run_std_mm", "chamfer_rmse_mean_mm",
        "chamfer_between_run_std_mm",
        "max_error_mean_across_runs_mm", "max_error_between_run_std_mm",
        "max_error_worst_run_mm",
    ]
    write_csv(output_dir / "centerline_metrics_summary.csv", method_rows, method_fields)

    print("\n[SUMMARY]")
    print(pd.DataFrame(method_rows).to_string(index=False))
    print(f"\n[DONE] {output_dir / 'centerline_metrics_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
