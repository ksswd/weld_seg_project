#!/usr/bin/env python3
"""Rigorous traditional baselines for weld point segmentation.

Protocol
--------
* Geometry features are converted to within-cloud percentile ranks.  This
  removes the invalid dependency on split-specific z-score standardizers.
* Hyperparameters are selected on validation only; test is evaluated once.
* RANSAC operates on the original PLY coordinates, never standardized XYZ.
* Metrics are accumulated from one global confusion matrix for every method.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io_utils import read_ply_with_scalar_seam


FEATURES = ("curvature", "density", "linearity")
QUERY_WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
SCORE_WEIGHTS = {
    "rank_curvature": (1.0, 0.0, 0.0),
    "rank_low_density": (0.0, 1.0, 0.0),
    "rank_linearity": (0.0, 0.0, 1.0),
    "rank_curvature_density": (1.0, 1.0, 0.0),
    "rank_density_linearity": (0.0, 1.0, 1.0),
    "rank_all_equal": (1.0, 1.0, 1.0),
    "rank_all_density2": (1.0, 2.0, 1.0),
    "rank_all_density3": (1.0, 3.0, 1.0),
}


def confusion(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    return np.array(
        [np.sum(y & pred), np.sum(~y & ~pred), np.sum(~y & pred), np.sum(y & ~pred)],
        dtype=np.int64,
    )


def metrics(c):
    tp, tn, fp, fn = map(float, c)
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    weld_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weld_iou": weld_iou,
        "background_iou": background_iou,
        "miou": (weld_iou + background_iou) / 2.0,
    }


def processed_files(root, split):
    return sorted((root / split).glob("*.csv"))


def raw_path_for(csv_path, raw_root, split):
    name = csv_path.name
    for suffix in ("_label_auto.csv", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)] + ".ply"
            break
    path = raw_root / split / name
    if not path.is_file():
        raise FileNotFoundError(f"raw PLY corresponding to {csv_path.name} was not found: {path}")
    return path


def percentile_rank(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    result = np.full(len(x), 0.5, dtype=np.float32)
    if np.any(finite):
        result[finite] = (rankdata(x[finite], method="average") - 0.5) / np.sum(finite)
    return result


def oriented_ranks(df):
    """High values consistently mean more weld-like."""
    curvature = percentile_rank(df["curvature"].to_numpy())
    low_density = 1.0 - percentile_rank(df["density"].to_numpy())
    linearity = percentile_rank(df["linearity"].to_numpy())
    return np.column_stack((curvature, low_density, linearity))


def weighted_score(ranks, weights):
    w = np.asarray(weights, dtype=np.float64)
    return ranks @ w / np.sum(w)


def sample_rank_data(files, max_points, seed):
    rng = np.random.default_rng(seed)
    per_file = max(1000, max_points // max(len(files), 1))
    scores = {name: [] for name in SCORE_WEIGHTS}
    ys = []
    for path in files:
        df = pd.read_csv(path, usecols=[*FEATURES, "label"])
        ranks = oriented_ranks(df)
        index = np.arange(len(df))
        if len(index) > per_file:
            index = rng.choice(index, per_file, replace=False)
        ys.append((df["label"].to_numpy()[index] > 0.5).astype(np.uint8))
        for name, weights in SCORE_WEIGHTS.items():
            scores[name].append(weighted_score(ranks[index], weights))
    return np.concatenate(ys), {name: np.concatenate(parts) for name, parts in scores.items()}


def choose_threshold(y, score, steps):
    # Dense coverage near the small positive ratios expected in weld data.
    positive_fractions = np.unique(
        np.r_[np.geomspace(0.002, 0.20, steps // 2), np.linspace(0.21, 0.60, steps // 2)]
    )
    thresholds = np.unique(np.quantile(score, 1.0 - positive_fractions))
    best = None
    for threshold in thresholds:
        result = metrics(confusion(y, score >= threshold))
        candidate = (result["miou"], result["f1"], float(threshold), result)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[3]


def tune_rank_methods(files, max_points, steps, seed):
    y, all_scores = sample_rank_data(files, max_points, seed)
    selected = {}
    candidates = {}
    for name, score in all_scores.items():
        threshold, result = choose_threshold(y, score, steps)
        candidates[name] = {
            "weights": list(SCORE_WEIGHTS[name]),
            "threshold": threshold,
            **{"sampled_validation_" + key: value for key, value in result.items()},
        }
    # Keep three true single-feature baselines and one selected interpretable fusion.
    for name in ("rank_curvature", "rank_low_density", "rank_linearity"):
        selected[name] = candidates[name]
    fusion_names = [name for name in candidates if name not in selected]
    best_fusion = max(
        fusion_names,
        key=lambda name: (candidates[name]["sampled_validation_miou"], candidates[name]["sampled_validation_f1"]),
    )
    selected["rank_feature_fusion"] = {"source_candidate": best_fusion, **candidates[best_fusion]}
    return selected, candidates


def plane_from_three(points):
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    return normal, -float(normal @ points[0])


def cloud_spacing(xyz, sample_cap, seed):
    rng = np.random.default_rng(seed)
    if len(xyz) > sample_cap:
        query = xyz[rng.choice(len(xyz), sample_cap, replace=False)]
    else:
        query = xyz
    tree = cKDTree(xyz)
    distance, _ = tree.query(query, k=2, workers=QUERY_WORKERS)
    spacing = float(np.median(distance[:, 1]))
    return max(spacing, float(np.linalg.norm(np.ptp(xyz, axis=0))) * 1e-7, 1e-9)


def ransac_plane(points, tolerance, iterations, sample_cap, seed):
    rng = np.random.default_rng(seed)
    fit = points if len(points) <= sample_cap else points[rng.choice(len(points), sample_cap, replace=False)]
    best = None
    for _ in range(iterations):
        model = plane_from_three(fit[rng.choice(len(fit), 3, replace=False)])
        if model is None:
            continue
        normal, offset = model
        inlier = np.abs(fit @ normal + offset) <= tolerance
        candidate = (int(np.sum(inlier)), normal, offset)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    _, normal, offset = best
    inlier = np.abs(fit @ normal + offset) <= tolerance
    if np.sum(inlier) >= 3:
        center = fit[inlier].mean(axis=0)
        _, _, vh = np.linalg.svd(fit[inlier] - center, full_matrices=False)
        normal = vh[-1]
        offset = -float(normal @ center)
    support = int(np.sum(np.abs(points @ normal + offset) <= tolerance))
    return normal, offset, support


def fit_dominant_planes(xyz, spacing, iterations, sample_cap, seed, fit_tolerance_mult, max_planes=2):
    tolerance = spacing * fit_tolerance_mult
    remaining = np.arange(len(xyz))
    planes = []
    for plane_index in range(max_planes):
        if len(remaining) < max(100, int(0.08 * len(xyz))):
            break
        model = ransac_plane(
            xyz[remaining], tolerance, iterations, sample_cap, seed + plane_index * 997
        )
        if model is None:
            break
        normal, offset, local_support = model
        distance_all = np.abs(xyz @ normal + offset)
        support = int(np.sum(distance_all <= tolerance))
        # Reject a second plane that is too small and likely represents the weld itself.
        if plane_index > 0 and support < int(0.10 * len(xyz)):
            break
        planes.append((normal, offset, support))
        remaining = remaining[distance_all[remaining] > tolerance]
    if not planes:
        center = xyz.mean(axis=0)
        _, _, vh = np.linalg.svd(xyz - center, full_matrices=False)
        planes = [(vh[-1], -float(vh[-1] @ center), len(xyz))]
    residual = np.column_stack([np.abs(xyz @ normal + offset) for normal, offset, _ in planes])
    return np.min(residual, axis=1) / spacing, planes


def load_raw(csv_path, raw_root, split, check_alignment=True):
    raw_path = raw_path_for(csv_path, raw_root, split)
    raw = read_ply_with_scalar_seam(str(raw_path))
    xyz = np.asarray(raw["xyz"], dtype=np.float64)
    if raw["scalar_label"] is None:
        raise ValueError(f"PLY has no weld label: {raw_path}")
    y = np.asarray(raw["scalar_label"]) > 0.5
    if check_alignment:
        csv_y = pd.read_csv(csv_path, usecols=["label"])["label"].to_numpy() > 0.5
        permutation = np.random.default_rng(42).permutation(len(y))
        if len(csv_y) != len(y) or not np.array_equal(csv_y, y[permutation]):
            raise ValueError(f"processed/raw label alignment check failed: {csv_path.name}")
    return raw_path, xyz, y


def tune_ransac(files, raw_root, iterations, sample_cap, seed, fit_tolerance_mult):
    residual_thresholds = np.array([0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])
    totals = {float(value): np.zeros(4, dtype=np.int64) for value in residual_thresholds}
    plane_counts = []
    for index, csv_path in enumerate(files):
        _, xyz, y = load_raw(csv_path, raw_root, "validation")
        spacing = cloud_spacing(xyz, sample_cap, seed + index)
        residual, planes = fit_dominant_planes(
            xyz, spacing, iterations, sample_cap, seed + index * 17, fit_tolerance_mult
        )
        plane_counts.append(len(planes))
        for threshold in residual_thresholds:
            totals[float(threshold)] += confusion(y, residual > threshold)
    best = max(totals, key=lambda value: (metrics(totals[value])["miou"], metrics(totals[value])["f1"]))
    return {
        "fit_tolerance_spacing_mult": float(fit_tolerance_mult),
        "residual_threshold_spacing_mult": float(best),
        "max_planes": 2,
        "validation_plane_count_mean": float(np.mean(plane_counts)),
        **{"validation_" + key: value for key, value in metrics(totals[best]).items()},
    }


def feature_prediction(csv_path, config):
    df = pd.read_csv(csv_path, usecols=[*FEATURES, "label"])
    ranks = oriented_ranks(df)
    score = weighted_score(ranks, config["weights"])
    pred_csv = score >= config["threshold"]
    permutation = np.random.default_rng(42).permutation(len(df))
    pred_raw = np.zeros(len(df), dtype=bool)
    pred_raw[permutation] = pred_csv
    return pred_raw


def evaluate(methods, files, raw_root, split, out_dir, iterations, sample_cap, seed, save_predictions):
    totals = {name: np.zeros(4, dtype=np.int64) for name in methods}
    times = {name: [] for name in methods}
    details = []
    for index, csv_path in enumerate(files):
        raw_path, xyz, y = load_raw(csv_path, raw_root, split)
        ransac_residual = None
        for name, config in methods.items():
            start = time.perf_counter()
            if name == "ransac_multiplane_residual":
                spacing = cloud_spacing(xyz, sample_cap, seed + index)
                ransac_residual, planes = fit_dominant_planes(
                    xyz,
                    spacing,
                    iterations,
                    sample_cap,
                    seed + index * 17,
                    config["fit_tolerance_spacing_mult"],
                )
                pred = ransac_residual > config["residual_threshold_spacing_mult"]
            else:
                pred = feature_prediction(csv_path, config)
            elapsed = time.perf_counter() - start
            c = confusion(y, pred)
            totals[name] += c
            times[name].append(elapsed)
            details.append(
                {
                    "method": name,
                    "file": raw_path.name,
                    "weld_type": int(re.match(r"^(\d+)", raw_path.name).group(1)),
                    "num_points": len(y),
                    "positive_ratio": float(np.mean(y)),
                    "infer_time_sec": elapsed,
                    "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
                    **metrics(c),
                }
            )
            if save_predictions:
                target = out_dir / "predictions" / name
                target.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                     "label": y.astype(np.uint8), "pred": pred.astype(np.uint8)}
                ).to_csv(target / (raw_path.stem + "_fullpred.csv"), index=False)
    detail_df = pd.DataFrame(details)
    detail_df.to_csv(out_dir / f"{split}_per_file_metrics.csv", index=False)
    type_rows = []
    for (method, weld_type), group in detail_df.groupby(["method", "weld_type"]):
        c = group[["tp", "tn", "fp", "fn"]].sum().to_numpy(dtype=np.int64)
        type_rows.append({
            "method": method,
            "weld_type": int(weld_type),
            "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
            "ground_truth_positive_ratio": float((c[0] + c[3]) / np.sum(c)),
            **metrics(c),
            "num_files": int(len(group)),
            "total_points": int(np.sum(c)),
        })
    pd.DataFrame(type_rows).sort_values(["method", "weld_type"]).to_csv(
        out_dir / f"{split}_per_type_metrics.csv", index=False
    )
    rows = []
    for name, c in totals.items():
        method_times = np.asarray(times[name])
        row = {
            "method": name,
            "split": split,
            "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
            "ground_truth_positive_ratio": float((c[0] + c[3]) / np.sum(c)),
            "predicted_positive_ratio": float((c[0] + c[2]) / np.sum(c)),
            **metrics(c),
            "num_files": len(files),
            "total_points": int(np.sum(c)),
            "mean_infer_ms": float(np.mean(method_times) * 1000.0),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("miou", ascending=False)


def separability_report(files, split):
    rows = []
    for csv_path in files:
        df = pd.read_csv(csv_path, usecols=[*FEATURES, "label"])
        ranks = oriented_ranks(df)
        y = df["label"].to_numpy() > 0.5
        for index, feature in enumerate(("curvature", "low_density", "linearity")):
            if np.unique(y).size < 2:
                continue
            rows.append({
                "split": split,
                "file": csv_path.name,
                "feature": feature,
                "roc_auc": roc_auc_score(y, ranks[:, index]),
                "average_precision": average_precision_score(y, ranks[:, index]),
                "positive_ratio": float(np.mean(y)),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="data/dataset_processed_1-17")
    parser.add_argument("--raw-root", default="data/dataset_1-17")
    parser.add_argument("--out-dir", default="results/traditional_baselines_v2")
    parser.add_argument("--threshold-steps", type=int, default=160)
    parser.add_argument("--tune-max-points", type=int, default=600000)
    parser.add_argument("--ransac-iterations", type=int, default=160)
    parser.add_argument("--ransac-sample-cap", type=int, default=30000)
    parser.add_argument("--ransac-fit-tolerance-mult", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0, help="smoke test only")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = processed_files(processed_root, "validation")
    test = processed_files(processed_root, "test")
    if args.limit_files:
        validation = validation[: args.limit_files]
        test = test[: args.limit_files]
    if not validation or not test:
        raise SystemExit("processed root must contain non-empty validation/ and test/ CSV directories")
    print(f"[DATA] validation={len(validation)} test={len(test)}")

    rank_methods, rank_candidates = tune_rank_methods(
        validation, args.tune_max_points, args.threshold_steps, args.seed
    )
    methods = dict(rank_methods)
    methods["ransac_multiplane_residual"] = tune_ransac(
        validation,
        raw_root,
        args.ransac_iterations,
        args.ransac_sample_cap,
        args.seed,
        args.ransac_fit_tolerance_mult,
    )
    parameters = {
        "protocol": {
            "selection_split": "validation",
            "final_split": "test",
            "feature_normalization": "within-cloud percentile rank",
            "ransac_coordinates": "original PLY XYZ",
            "global_confusion_matrix": True,
            "seed": args.seed,
        },
        "selected_methods": methods,
        "rank_fusion_candidates": rank_candidates,
    }
    (out_dir / "selected_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation_summary = evaluate(
        methods, validation, raw_root, "validation", out_dir,
        args.ransac_iterations, args.ransac_sample_cap, args.seed, False,
    )
    test_summary = evaluate(
        methods, test, raw_root, "test", out_dir,
        args.ransac_iterations, args.ransac_sample_cap, args.seed + 10000,
        args.save_predictions,
    )
    validation_summary.to_csv(out_dir / "validation_summary.csv", index=False)
    test_summary.to_csv(out_dir / "traditional_baselines_v2_summary.csv", index=False)
    separability = pd.concat(
        [separability_report(validation, "validation"), separability_report(test, "test")],
        ignore_index=True,
    )
    separability.to_csv(out_dir / "feature_separability_per_file.csv", index=False)
    separability.groupby(["split", "feature"], as_index=False)[
        ["roc_auc", "average_precision", "positive_ratio"]
    ].mean().to_csv(out_dir / "feature_separability_summary.csv", index=False)

    print("\n[VALIDATION]\n" + validation_summary.to_string(index=False))
    print("\n[TEST]\n" + test_summary.to_string(index=False))
    print(f"\n[DONE] {out_dir / 'traditional_baselines_v2_summary.csv'}")


if __name__ == "__main__":
    main()
