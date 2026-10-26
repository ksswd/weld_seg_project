#!/usr/bin/env python3
"""Tune traditional weld baselines on validation and evaluate once on test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURES = ("curvature", "density", "linearity")


def confusion(y, p):
    y = np.asarray(y, dtype=bool); p = np.asarray(p, dtype=bool)
    return np.array([np.sum(y & p), np.sum(~y & ~p), np.sum(~y & p), np.sum(y & ~p)], dtype=np.int64)


def metric_row(c):
    tp, tn, fp, fn = map(float, c); eps = 1e-12
    precision = tp / (tp + fp + eps); recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    wiou = tp / (tp + fp + fn + eps); biou = tn / (tn + fp + fn + eps)
    return dict(accuracy=(tp + tn) / max(tp + tn + fp + fn, 1), precision=precision,
                recall=recall, f1=f1, weld_iou=wiou, background_iou=biou, miou=(wiou + biou) / 2)


def csv_files(path):
    return sorted(p for p in Path(path).glob("*.csv") if "pred" not in p.name)


def sample_validation(files, max_points, seed):
    rng = np.random.default_rng(seed); chunks = []
    per_file = max(1000, max_points // max(len(files), 1))
    for path in files:
        df = pd.read_csv(path, usecols=[*FEATURES, "label"])
        if len(df) > per_file:
            df = df.iloc[rng.choice(len(df), per_file, replace=False)]
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def tune_feature_thresholds(files, steps, max_points, seed):
    df = sample_validation(files, max_points, seed); y = df.label.to_numpy() > .5
    configs = {}
    for feature in FEATURES:
        x = df[feature].to_numpy(dtype=np.float64)
        thresholds = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(.002, .998, steps)))
        best = None
        for direction in ("ge", "le"):
            for threshold in thresholds:
                pred = x >= threshold if direction == "ge" else x <= threshold
                score = metric_row(confusion(y, pred))
                candidate = (score["miou"], score["f1"], float(threshold), direction)
                if best is None or candidate[:2] > best[:2]: best = candidate
        configs[f"threshold_{feature}"] = {"feature": feature, "threshold": best[2], "direction": best[3],
                                            "validation_miou": best[0], "validation_f1": best[1]}
    return configs


def polynomial_geometry_features(x):
    """Degree-2 terms for curvature, density and linearity (no bias)."""
    return PolynomialFeatures(2, include_bias=False).fit_transform(x)


def tune_multifeature_score(files, steps, max_points, seed):
    """Fit an interpretable quadratic geometric score on validation only."""
    df = sample_validation(files, max_points, seed).replace([np.inf, -np.inf], np.nan).dropna()
    x = df[list(FEATURES)].to_numpy(dtype=np.float64)
    y = (df.label.to_numpy() > .5).astype(np.uint8)
    phi = polynomial_geometry_features(x)
    scaler = StandardScaler().fit(phi)
    phi_scaled = scaler.transform(phi)
    model = LogisticRegression(class_weight="balanced", C=.3, max_iter=500, random_state=seed)
    model.fit(phi_scaled, y)

    # Fold standardization into the coefficients so evaluation only needs a
    # transparent polynomial score and a threshold.
    coef = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_))
    score = phi @ coef + intercept
    thresholds = np.unique(np.quantile(score[np.isfinite(score)], np.linspace(.01, .99, steps)))
    best = None
    for threshold in thresholds:
        result = metric_row(confusion(y, score >= threshold))
        candidate = (result["miou"], result["f1"], float(threshold))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return {
        "features": list(FEATURES),
        "degree": 2,
        "coefficients": coef.tolist(),
        "intercept": intercept,
        "threshold": best[2],
        "validation_miou": best[0],
        "validation_f1": best[1],
    }


def plane_from_three(pts):
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0]); norm = np.linalg.norm(normal)
    if norm < 1e-12: return None
    normal /= norm
    return normal, -float(normal @ pts[0])


def fit_plane(points, iterations, sample_cap, seed):
    rng = np.random.default_rng(seed)
    fit = points if len(points) <= sample_cap else points[rng.choice(len(points), sample_cap, replace=False)]
    scale = max(float(np.linalg.norm(np.ptp(fit, axis=0))), 1e-8)
    provisional = scale * .003
    best = None
    for _ in range(iterations):
        model = plane_from_three(fit[rng.choice(len(fit), 3, replace=False)])
        if model is None: continue
        n, d = model; residual = np.abs(fit @ n + d); count = int(np.sum(residual <= provisional))
        if best is None or count > best[0]: best = (count, n, d)
    if best is None: return np.array([0., 0., 1.]), 0., scale
    _, n, d = best
    inlier = np.abs(fit @ n + d) <= provisional
    if np.sum(inlier) >= 3:
        center = fit[inlier].mean(0); _, _, vh = np.linalg.svd(fit[inlier] - center, full_matrices=False)
        n = vh[-1]; d = -float(n @ center)
    return n, d, scale


def geometry_cache(path, iterations, sample_cap, seed):
    df = pd.read_csv(path, usecols=["x", "y", "z", "label"])
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64); y = df.label.to_numpy() > .5
    n1, d1, scale = fit_plane(xyz, iterations, sample_cap, seed)
    r1 = np.abs(xyz @ n1 + d1)
    keep = r1 > scale * .004
    second = xyz[keep]
    if len(second) >= 3:
        n2, d2, _ = fit_plane(second, iterations, sample_cap, seed + 1)
        r2 = np.abs(xyz @ n2 + d2)
    else:
        n2, d2, r2 = n1, d1, np.full(len(xyz), np.inf)
    return df, xyz, y, r1 / scale, r2 / scale


def single_plane_residual(df, iterations, sample_cap, seed):
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    n, d, scale = fit_plane(xyz, iterations, sample_cap, seed)
    return np.abs(xyz @ n + d) / scale


def tune_geometry(files, iterations, sample_cap, seed):
    single_grid = np.array([.0005, .001, .002, .003, .005, .008, .012, .02])
    corridor_grid = np.array([.001, .002, .003, .005, .008, .012, .02, .03])
    conf_single = {v: np.zeros(4, dtype=np.int64) for v in single_grid}
    conf_multi = {v: np.zeros(4, dtype=np.int64) for v in corridor_grid}
    for i, path in enumerate(files):
        _, _, y, r1, r2 = geometry_cache(path, iterations, sample_cap, seed + i * 2)
        for v in single_grid: conf_single[v] += confusion(y, r1 > v)
        for v in corridor_grid: conf_multi[v] += confusion(y, (r1 <= v) & (r2 <= v))
    def choose(table):
        return max(table, key=lambda v: (metric_row(table[v])["miou"], metric_row(table[v])["f1"]))
    s = choose(conf_single); m = choose(conf_multi)
    return {
        "ransac_single_plane_residual": {"distance_ratio": float(s), **{"validation_" + k: v for k, v in metric_row(conf_single[s]).items()}},
        "ransac_two_plane_intersection": {"corridor_ratio": float(m), **{"validation_" + k: v for k, v in metric_row(conf_multi[m]).items()}},
    }


def evaluate(methods, files, out_dir, iterations, sample_cap, seed, save_predictions):
    rows = []; totals = {name: np.zeros(4, dtype=np.int64) for name in methods}
    for i, path in enumerate(files):
        df_all = pd.read_csv(path)
        y = df_all.label.to_numpy() > .5
        for name, cfg in methods.items():
            start = time.perf_counter()
            if name.startswith("threshold_"):
                x = df_all[cfg["feature"]].to_numpy()
                pred = x >= cfg["threshold"] if cfg["direction"] == "ge" else x <= cfg["threshold"]
            elif name == "multi_feature_geometric_score":
                x = df_all[cfg["features"]].replace([np.inf, -np.inf], 0).to_numpy(dtype=np.float64)
                phi = polynomial_geometry_features(x)
                score = phi @ np.asarray(cfg["coefficients"]) + cfg["intercept"]
                pred = score >= cfg["threshold"]
            else:
                if name == "ransac_single_plane_residual":
                    r1 = single_plane_residual(df_all, iterations, sample_cap, seed + i * 2)
                    pred = r1 > cfg["distance_ratio"]
                else:
                    _, _, _, r1, r2 = geometry_cache(path, iterations, sample_cap, seed + i * 2)
                    pred = (r1 <= cfg["corridor_ratio"]) & (r2 <= cfg["corridor_ratio"])
            elapsed = time.perf_counter() - start; c = confusion(y, pred); totals[name] += c
            rows.append({"method": name, "file": path.name, "num_points": len(y), "infer_time_sec": elapsed,
                         **metric_row(c)})
            if save_predictions:
                target = out_dir / "predictions" / name; target.mkdir(parents=True, exist_ok=True)
                out = df_all[["x", "y", "z", "label"]].copy(); out["pred"] = pred.astype(np.uint8)
                out.to_csv(target / path.name.replace(".csv", "_fullpred.csv"), index=False)
    detail = pd.DataFrame(rows); detail.to_csv(out_dir / "per_file_metrics.csv", index=False)
    summary = []
    for name, c in totals.items():
        times = detail.loc[detail.method == name, "infer_time_sec"]
        points = detail.loc[detail.method == name, "num_points"].sum()
        summary.append({"method": name, **metric_row(c), "num_files": int(len(times)), "total_points": int(points),
                        "total_infer_sec": float(times.sum()), "mean_infer_ms": float(times.mean() * 1000),
                        "throughput_points_sec": float(points / max(times.sum(), 1e-12))})
    pd.DataFrame(summary).sort_values("miou", ascending=False).to_csv(out_dir / "traditional_baselines_summary.csv", index=False)
    print(pd.DataFrame(summary).sort_values("miou", ascending=False).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/dataset_processed_1-17")
    ap.add_argument("--out-dir", default="results/traditional_baselines")
    ap.add_argument("--threshold-steps", type=int, default=201)
    ap.add_argument("--tune-max-points", type=int, default=500000)
    ap.add_argument("--ransac-iterations", type=int, default=120)
    ap.add_argument("--ransac-sample-cap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--limit-files", type=int, default=0, help="smoke-test only; 0 uses every file")
    args = ap.parse_args(); root = Path(args.data_root); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    val, test = csv_files(root / "validation"), csv_files(root / "test")
    if args.limit_files > 0:
        val, test = val[:args.limit_files], test[:args.limit_files]
    if not val or not test: raise SystemExit("data root must contain non-empty validation/ and test/ CSV directories")
    print(f"[DATA] validation={len(val)} test={len(test)}")
    configs = tune_feature_thresholds(val, args.threshold_steps, args.tune_max_points, args.seed)
    configs["multi_feature_geometric_score"] = tune_multifeature_score(
        val, args.threshold_steps, args.tune_max_points, args.seed
    )
    configs.update(tune_geometry(val, args.ransac_iterations, args.ransac_sample_cap, args.seed))
    (out / "selected_parameters.json").write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(configs, ensure_ascii=False, indent=2))
    evaluate(configs, test, out, args.ransac_iterations, args.ransac_sample_cap, args.seed + 10000, args.save_predictions)


if __name__ == "__main__": main()
