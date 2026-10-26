import argparse
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0]), 0.0
    normal = normal / norm
    d = -np.dot(normal, p1)
    return normal, d


def point_plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.abs(points @ normal + d)


def ransac_plane(points: np.ndarray, iters: int, distance_thr: float) -> Tuple[np.ndarray, float, np.ndarray]:
    best_inliers = None
    best_normal = np.array([0.0, 0.0, 0.0])
    best_d = 0.0
    n_points = points.shape[0]
    rng = np.random.default_rng(42)

    for _ in range(iters):
        idx = rng.choice(n_points, 3, replace=False)
        normal, d = plane_from_points(points[idx[0]], points[idx[1]], points[idx[2]])
        if np.allclose(normal, 0.0):
            continue
        distances = point_plane_distance(points, normal, d)
        inliers = distances < distance_thr
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if best_inliers is None:
        best_inliers = np.zeros(n_points, dtype=bool)
    return best_normal, best_d, best_inliers


def cluster_weld(points: np.ndarray, mask: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    remaining = points[mask]
    if remaining.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=np.int32)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(remaining)
    pred = np.zeros(points.shape[0], dtype=np.int32)
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() < min_samples:
            continue
        pred_indices = np.where(mask)[0][cluster_mask]
        pred[pred_indices] = 1
    return pred


def process_file(
    path: str,
    output_dir: str,
    iters: int,
    distance_thr: float,
    eps: float,
    min_samples: int,
) -> dict:
    data = pd.read_csv(path)
    points = data[["x", "y", "z"]].to_numpy(dtype=np.float32)
    labels = data["label"].to_numpy(dtype=np.int32)

    start = time.perf_counter()
    _, _, inliers = ransac_plane(points, iters, distance_thr)
    remaining_mask = ~inliers
    pred = cluster_weld(points, remaining_mask, eps, min_samples)
    duration = time.perf_counter() - start

    out = data[["x", "y", "z", "label"]].copy()
    out["pred"] = pred

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(path).replace(".csv", "_pred_method2.csv"))
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
        "ransac_iters": iters,
        "distance_thr": distance_thr,
        "eps": eps,
        "min_samples": min_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "seconds": duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Method2: RANSAC plane removal + DBSCAN clustering")
    parser.add_argument("--input_dir", default="comparable_methods/mathmatic/data/raw")
    parser.add_argument("--output_dir", default="comparable_methods/mathmatic/data/preds_method2")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--distance_thr", type=float, default=0.3)
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument("--min_samples", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".csv")])
    if args.limit > 0:
        files = files[: args.limit]

    results = []
    for name in files:
        path = os.path.join(args.input_dir, name)
        results.append(
            process_file(
                path,
                args.output_dir,
                args.iters,
                args.distance_thr,
                args.eps,
                args.min_samples,
            )
        )

    summary = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir, "summary_method2.csv"), index=False)


if __name__ == "__main__":
    main()
