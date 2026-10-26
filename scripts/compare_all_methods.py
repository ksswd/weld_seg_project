#!/usr/bin/env python3
"""Audit and summarize weld-segmentation benchmark outputs.

The script deliberately recomputes every metric from per-point prediction files
instead of trusting method-specific logs.  Timing is reported only when a timing
CSV is present; its comparability is audited from the recorded point counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_METHODS = {
    "ours": "results/final/seed_points_sweep",
    "dgcnn": "comparable_methods/dgcnn.pytorch/outputs/seed_points_sweep",
    "pointcept": "comparable_methods/Pointcept-copy/exp/weld_1_17/seed_points_sweep",
    "pointmlp": "comparable_methods/pointMLP-pytorch/part_segmentation/outputs/seed_points_sweep",
    "pointnet2": "comparable_methods/Pointnet_Pointnet2_pytorch/fullpred/seed_points_sweep",
    "pointconv": "comparable_methods/pointconv_pytorch/log/seed_points_sweep",
}


@dataclass
class Prediction:
    path: Path
    label: np.ndarray
    pred: np.ndarray
    prob: np.ndarray | None


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must be NAME=PATH")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def choose_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def read_prediction(path: Path, threshold: float) -> Prediction:
    df = pd.read_csv(path)
    label_col = choose_column(df, ("label", "gt", "target", "scalar_label"))
    pred_col = choose_column(df, ("pred", "prediction"))
    prob_col = choose_column(df, ("prob", "probability", "weld_prob"))
    if label_col is None:
        raise ValueError("missing label/gt column")
    if pred_col is None and prob_col is None:
        raise ValueError("missing pred or prob column")
    label = (df[label_col].to_numpy(dtype=np.float64) > 0.5).astype(np.uint8)
    prob = df[prob_col].to_numpy(dtype=np.float64) if prob_col else None
    pred = ((prob > threshold) if pred_col is None else
            (df[pred_col].to_numpy(dtype=np.float64) > 0.5)).astype(np.uint8)
    if label.size != pred.size:
        raise ValueError(f"label/pred length mismatch: {label.size}/{pred.size}")
    return Prediction(path, label, pred, prob)


def evaluate_prediction_file(path: Path, threshold: float) -> tuple[int, int, int, int, int, str]:
    """Evaluate a CSV incrementally so a large benchmark cannot exhaust RAM."""
    if path.suffix.lower() == ".txt":
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 5:
            raise ValueError("text prediction needs x y z label pred columns")
        label = (arr[:, 3] > 0.5).astype(np.uint8)
        pred = (arr[:, 4] > 0.5).astype(np.uint8)
        gt_signature = hashlib.sha256(f"{label.size}:{int(label.sum())}".encode("ascii")).hexdigest()
        return (*confusion(label, pred), int(label.size), gt_signature)
    tp = tn = fp = fn = count = 0
    label_positive = 0
    for df in pd.read_csv(path, chunksize=250_000):
        label_col = choose_column(df, ("label", "gt", "target", "scalar_label"))
        pred_col = choose_column(df, ("pred", "prediction"))
        prob_col = choose_column(df, ("prob", "probability", "weld_prob"))
        if label_col is None or (pred_col is None and prob_col is None):
            raise ValueError("missing label and/or prediction column")
        label = (df[label_col].to_numpy(dtype=np.float64) > 0.5).astype(np.uint8)
        label_positive += int(label.sum())
        source = df[pred_col] if pred_col else df[prob_col]
        pred = (source.to_numpy(dtype=np.float64) > (0.5 if pred_col else threshold)).astype(np.uint8)
        c = confusion(label, pred)
        tp += c[0]; tn += c[1]; fp += c[2]; fn += c[3]; count += label.size
    gt_signature = hashlib.sha256(f"{count}:{label_positive}".encode("ascii")).hexdigest()
    return tp, tn, fp, fn, count, gt_signature


def confusion(label: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((label == 1) & (pred == 1)))
    tn = int(np.sum((label == 0) & (pred == 0)))
    fp = int(np.sum((label == 0) & (pred == 1)))
    fn = int(np.sum((label == 1) & (pred == 0)))
    return tp, tn, fp, fn


def metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    eps = 1e-12
    total = tp + tn + fp + fn
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    weld_iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    return {
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weld_iou": weld_iou,
        "background_iou": bg_iou,
        "miou": (weld_iou + bg_iou) / 2,
    }


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def prediction_files(run_dir: Path) -> list[Path]:
    files = list(run_dir.rglob("*.csv"))
    selected = [
        p for p in files
        if ("fullpred" in p.name.lower() or "fullpred" in str(p.parent).lower())
        and "timing" not in p.name.lower()
        and "metric" not in p.name.lower()
        and "summary" not in p.name.lower()
    ]
    if not selected:
        selected = [p for p in run_dir.rglob("*.txt") if "pred" in str(p.parent).lower()]
    return sorted(selected)


def timing_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.rglob("*.csv") if "tim" in p.name.lower())


def read_timing(files: list[Path]) -> dict[str, float | str]:
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        time_col = choose_column(df, ("model_forward_sec", "infer_time_sec", "inference_time_sec",
                                      "estimated_file_infer_time_sec", "time_sec"))
        points_col = choose_column(df, ("num_points", "full_points", "valid_points", "points"))
        if time_col is None:
            continue
        small = pd.DataFrame({"seconds": pd.to_numeric(df[time_col], errors="coerce")})
        small["points"] = pd.to_numeric(df[points_col], errors="coerce") if points_col else np.nan
        frames.append(small.dropna(subset=["seconds"]))
    if not frames:
        return {"timing_status": "missing"}
    df = pd.concat(frames, ignore_index=True)
    total_sec = float(df["seconds"].sum())
    total_points = float(df["points"].sum()) if df["points"].notna().all() else np.nan
    return {
        "timing_status": "recorded",
        "timed_samples": int(len(df)),
        "total_infer_sec": total_sec,
        "mean_infer_ms": float(df["seconds"].mean() * 1000),
        "throughput_points_sec": total_points / total_sec if total_sec > 0 and np.isfinite(total_points) else np.nan,
    }


def evaluate_run(method: str, run_dir: Path, threshold: float, chunk_size: int) -> dict:
    files = prediction_files(run_dir)
    if not files:
        raise ValueError("no full-prediction CSV files")
    tp = tn = fp = fn = total_points = 0
    errors = []
    file_sizes = []
    gt_hasher = hashlib.sha256()
    for path in files:
        try:
            c = evaluate_prediction_file(path, threshold)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        tp += c[0]; tn += c[1]; fp += c[2]; fn += c[3]
        total_points += c[4]
        file_sizes.append(c[4])
        canonical_name = re.sub(r"(_fullpred|_pred_xyz_label)$", "", path.stem)
        canonical_name = re.sub(r"_label_auto$", "", canonical_name)
        gt_hasher.update(canonical_name.encode("utf-8"))
        gt_hasher.update(str(c[4]).encode("ascii"))
        gt_hasher.update(c[5].encode("ascii"))
    if not file_sizes:
        raise ValueError("all prediction CSV files were invalid")
    row = {
        "method": method,
        "seed": seed_from_path(run_dir),
        "run_dir": str(run_dir),
        "num_files": len(file_sizes),
        "total_points": total_points,
        "mean_points_per_cloud": float(np.mean(file_sizes)),
        "chunk_size": chunk_size,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "gt_positive": tp + fn,
        "gt_negative": tn + fp,
        "gt_positive_ratio": (tp + fn) / max(total_points, 1),
        "gt_fingerprint": gt_hasher.hexdigest(),
        **metrics(tp, tn, fp, fn),
        **read_timing(timing_files(run_dir)),
        "prediction_errors": len(errors),
    }
    row["fair_timing"] = bool(
        row.get("timing_status") == "recorded"
        and np.isfinite(row.get("throughput_points_sec", np.nan))
        and row.get("timed_samples") == row["num_files"]
    )
    return row


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates = sorted({p.parent for p in root.rglob("eval_summary_metrics.csv")})
    if candidates:
        return candidates
    fullpred_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and "fullpred" in p.name.lower())
    if fullpred_dirs:
        return sorted({p.parent for p in fullpred_dirs})
    candidates = sorted({p.parent.parent if p.parent.name == "fullpred_test" else p.parent
                         for p in root.rglob("*fullpred.csv")})
    if not candidates:
        pred_dirs = sorted(p for p in root.rglob("pred_xyz_label_test") if p.is_dir())
        candidates = sorted({p.parent for p in pred_dirs})
    return candidates or [root]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and compare all weld segmentation results")
    parser.add_argument("--method", action="append", type=parse_method, default=[], metavar="NAME=PATH")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=Path("results/unified_benchmark"))
    parser.add_argument("--strict-timing", action="store_true", help="fail if timing is absent/not auditable")
    parser.add_argument("--strict-protocol", action="store_true",
                        help="fail unless every run has identical files, points and GT positives")
    parser.add_argument("--protocol-seed", type=int, default=20260823)
    args = parser.parse_args()
    methods = dict(args.method) if args.method else {k: Path(v) for k, v in DEFAULT_METHODS.items()}
    rows, failures = [], []
    for method, root in methods.items():
        for run_dir in discover_runs(root):
            try:
                rows.append(evaluate_run(method, run_dir, args.threshold, args.chunk_size))
            except Exception as exc:
                failures.append({"method": method, "run_dir": str(run_dir), "error": str(exc)})
    if not rows:
        raise SystemExit("No valid benchmark runs found")
    detail = pd.DataFrame(rows).sort_values(["method", "seed"], na_position="last")
    requested_methods = set(methods)
    observed_methods = set(detail["method"])
    missing_methods = sorted(requested_methods - observed_methods)
    numeric_keys = ["num_files", "total_points", "gt_positive", "gt_negative"]
    inconsistent = {key: sorted(detail[key].astype(int).unique().tolist()) for key in numeric_keys
                    if detail[key].nunique() != 1}
    if detail["gt_fingerprint"].nunique() != 1:
        inconsistent["gt_fingerprint"] = sorted(detail["gt_fingerprint"].unique().tolist())
    if args.strict_protocol and (failures or missing_methods or inconsistent):
        raise SystemExit(
            "Protocol audit failed: " + json.dumps(
                {"failures": failures, "missing_methods": missing_methods,
                 "inconsistent": inconsistent}, ensure_ascii=False))
    numeric = ["accuracy", "precision", "recall", "f1", "weld_iou", "background_iou", "miou",
               "mean_infer_ms", "throughput_points_sec"]
    for column in numeric:
        if column not in detail.columns:
            detail[column] = np.nan
    summary = detail.groupby("method", as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = ["method" if a == "method" else f"{a}_{b}" for a, b in summary.columns]
    fair = detail.groupby("method")["fair_timing"].all()
    summary["fair_timing"] = summary["method"].map(fair).fillna(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "benchmark_runs.csv", index=False)
    summary.to_csv(args.out_dir / "benchmark_summary.csv", index=False)
    (args.out_dir / "benchmark_audit.json").write_text(
        json.dumps({"protocol": {"threshold": args.threshold, "chunk_size": args.chunk_size,
                                  "protocol_seed": args.protocol_seed},
                    "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_dir.resolve()}")
    if args.strict_timing and not bool(detail["fair_timing"].all()):
        raise SystemExit("Timing audit failed: missing or non-comparable timing records")


if __name__ == "__main__":
    main()
