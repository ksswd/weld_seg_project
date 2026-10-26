import argparse
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_normals_curvature(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(points)
    _, indices = nn.kneighbors(points)

    normals = np.zeros_like(points)
    curvature = np.zeros(points.shape[0], dtype=np.float32)

    for i, neigh_idx in enumerate(indices):
        neigh = points[neigh_idx]
        centroid = neigh.mean(axis=0)
        cov = np.cov((neigh - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        normals[i] = eigvecs[:, 0]
        curvature[i] = eigvals[0] / (eigvals.sum() + 1e-12)

    return normals, curvature


def compute_normal_variation(normals: np.ndarray, points: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(points)
    _, indices = nn.kneighbors(points)

    variation = np.zeros(points.shape[0], dtype=np.float32)
    for i, neigh_idx in enumerate(indices):
        neigh_normals = normals[neigh_idx]
        dot = np.abs(neigh_normals @ normals[i])
        variation[i] = 1.0 - dot.mean()
    return variation


def predict_weld(points: np.ndarray, k: int, curv_thr: float, var_thr: float) -> np.ndarray:
    normals, curvature = compute_normals_curvature(points, k)
    variation = compute_normal_variation(normals, points, k)
    return ((curvature >= curv_thr) & (variation >= var_thr)).astype(np.int32)


def process_file(path: str, output_dir: str, k: int, curv_thr: float, var_thr: float) -> dict:
    data = pd.read_csv(path)
    points = data[["x", "y", "z"]].to_numpy(dtype=np.float32)
    labels = data["label"].to_numpy(dtype=np.int32)

    start = time.perf_counter()
    pred = predict_weld(points, k, curv_thr, var_thr)
    duration = time.perf_counter() - start

    out = data[["x", "y", "z", "label"]].copy()
    out["pred"] = pred

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(path).replace(".csv", "_pred_method1.csv"))
    out.to_csv(out_path, index=False)

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "file": os.path.basename(path),
        "points": len(points),
        "k": k,
        "curv_thr": curv_thr,
        "var_thr": var_thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "seconds": duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Method1: curvature + normal variation threshold")
    parser.add_argument("--input_dir", default="comparable_methods/mathmatic/data/raw")
    parser.add_argument("--output_dir", default="comparable_methods/mathmatic/data/preds_method1")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--curv_thr", type=float, default=0.02)
    parser.add_argument("--var_thr", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".csv")])
    if args.limit > 0:
        files = files[: args.limit]

    results = []
    for name in files:
        path = os.path.join(args.input_dir, name)
        results.append(process_file(path, args.output_dir, args.k, args.curv_thr, args.var_thr))

    summary = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir, "summary_method1.csv"), index=False)


if __name__ == "__main__":
    main()
