#!/usr/bin/env python3
"""Exact validation-only threshold sweep for three geometric features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


FEATURES = ("curvature", "density", "linearity")


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
    f1 = 2 * precision * recall / (precision + recall + eps)
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


def percentile_rank(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    result = np.full(len(x), 0.5, dtype=np.float32)
    if np.any(finite):
        result[finite] = (rankdata(x[finite], method="average") - 0.5) / np.sum(finite)
    return result


def files(root, split):
    return sorted((root / split).glob("*.csv"))


def load_feature(files_in, feature, representation):
    score_parts, label_parts = [], []
    for path in files_in:
        df = pd.read_csv(path, usecols=[feature, "label"])
        score = df[feature].to_numpy(dtype=np.float32)
        if representation == "within_cloud_percentile":
            score = percentile_rank(score)
        score_parts.append(score)
        label_parts.append((df["label"].to_numpy() > 0.5).astype(np.uint8))
    return np.concatenate(score_parts), np.concatenate(label_parts)


def candidate_at_start(sorted_score, cumulative_positive, total_positive, start):
    previous_positive = np.where(start > 0, cumulative_positive[start - 1], 0)
    tp = total_positive - previous_positive
    predicted = len(sorted_score) - start
    fp = predicted - tp
    fn = total_positive - tp
    tn = start - previous_positive
    return tp, tn, fp, fn


def candidate_at_end(cumulative_positive, total_positive, total_negative, end):
    tp = cumulative_positive[end]
    predicted = end + 1
    fp = predicted - tp
    fn = total_positive - tp
    tn = total_negative - fp
    return tp, tn, fp, fn


def best_from_candidates(score, y, direction, chunk_size=500000):
    finite = np.isfinite(score)
    if not np.all(finite):
        score = np.where(finite, score, np.nanmedian(score[finite]))
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order].astype(np.int64, copy=False)
    cumulative_positive = np.cumsum(sorted_y, dtype=np.int64)
    total_positive = int(cumulative_positive[-1])
    total_negative = int(len(y) - total_positive)
    changes = np.flatnonzero(sorted_score[1:] != sorted_score[:-1])
    indices = np.r_[0, changes + 1] if direction == "ge" else np.r_[changes, len(y) - 1]

    best = None
    for offset in range(0, len(indices), chunk_size):
        current = indices[offset : offset + chunk_size]
        if direction == "ge":
            tp, tn, fp, fn = candidate_at_start(
                sorted_score, cumulative_positive, total_positive, current
            )
        else:
            tp, tn, fp, fn = candidate_at_end(
                cumulative_positive, total_positive, total_negative, current
            )
        eps = 1e-12
        weld_iou = tp / (tp + fp + fn + eps)
        background_iou = tn / (tn + fp + fn + eps)
        miou = (weld_iou + background_iou) / 2.0
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        # Lexicographic objective: mIoU first, F1 only resolves ties.
        local = int(np.argmax(miou + f1 * 1e-12))
        index = int(current[local])
        c = np.array([tp[local], tn[local], fp[local], fn[local]], dtype=np.int64)
        candidate = (float(miou[local]), float(f1[local]), float(sorted_score[index]), c)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], metrics(best[3]), best[3]


def tune_feature(validation_files, feature):
    candidates = []
    for representation in ("absolute_value", "within_cloud_percentile"):
        score, y = load_feature(validation_files, feature, representation)
        for direction in ("ge", "le"):
            threshold, result, c = best_from_candidates(score, y, direction)
            candidates.append({
                "feature": feature,
                "representation": representation,
                "direction": direction,
                "threshold": threshold,
                "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
                **{"validation_" + key: value for key, value in result.items()},
            })
        del score, y
    selected = max(
        candidates,
        key=lambda item: (item["validation_miou"], item["validation_f1"]),
    )
    return selected, candidates


def predict_file(path, config):
    df = pd.read_csv(path, usecols=[config["feature"], "label"])
    score = df[config["feature"]].to_numpy(dtype=np.float32)
    if config["representation"] == "within_cloud_percentile":
        score = percentile_rank(score)
    pred = score >= config["threshold"] if config["direction"] == "ge" else score <= config["threshold"]
    return df["label"].to_numpy() > 0.5, pred


def evaluate(selected, test_files, out_dir):
    totals = {name: np.zeros(4, dtype=np.int64) for name in selected}
    type_totals = {}
    detail = []
    for path in test_files:
        match = re.match(r"^(\d+)", path.name)
        weld_type = int(match.group(1)) if match else -1
        for name, config in selected.items():
            y, pred = predict_file(path, config)
            c = confusion(y, pred)
            totals[name] += c
            type_totals.setdefault((name, weld_type), np.zeros(4, dtype=np.int64))
            type_totals[(name, weld_type)] += c
            detail.append({
                "method": name, "file": path.name, "weld_type": weld_type,
                "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
                "ground_truth_positive_ratio": float(np.mean(y)),
                "predicted_positive_ratio": float(np.mean(pred)),
                **metrics(c),
            })
    pd.DataFrame(detail).to_csv(out_dir / "threshold_sweep_test_per_file.csv", index=False)
    type_rows = []
    for (name, weld_type), c in sorted(type_totals.items()):
        type_rows.append({
            "method": name, "weld_type": weld_type,
            "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
            "ground_truth_positive_ratio": float((c[0] + c[3]) / np.sum(c)),
            **metrics(c),
        })
    pd.DataFrame(type_rows).to_csv(out_dir / "threshold_sweep_test_per_type.csv", index=False)
    rows = []
    for name, c in totals.items():
        rows.append({
            "method": name,
            "feature": selected[name]["feature"],
            "representation": selected[name]["representation"],
            "direction": selected[name]["direction"],
            "threshold": selected[name]["threshold"],
            "tp": int(c[0]), "tn": int(c[1]), "fp": int(c[2]), "fn": int(c[3]),
            "ground_truth_positive_ratio": float((c[0] + c[3]) / np.sum(c)),
            "predicted_positive_ratio": float((c[0] + c[2]) / np.sum(c)),
            **metrics(c),
            "num_files": len(test_files),
            "total_points": int(np.sum(c)),
        })
    return pd.DataFrame(rows).sort_values("miou", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="data/dataset_processed_1-17")
    parser.add_argument("--out-dir", default="results/traditional_threshold_sweep")
    parser.add_argument("--ransac-summary", default="")
    parser.add_argument("--limit-files", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = files(root, "validation")
    test = files(root, "test")
    if args.limit_files:
        validation, test = validation[: args.limit_files], test[: args.limit_files]
    if not validation or not test:
        raise SystemExit("processed root must contain validation/ and test/")
    print(f"[DATA] validation={len(validation)} test={len(test)}")

    selected, all_candidates = {}, {}
    for feature in FEATURES:
        print(f"[SWEEP] feature={feature} representations=2 directions=2 all_validation_points=true")
        best, candidates = tune_feature(validation, feature)
        selected[f"threshold_{feature}_best"] = best
        all_candidates[feature] = candidates
        print(json.dumps(best, ensure_ascii=False))
    (out_dir / "selected_thresholds.json").write_text(
        json.dumps({"selected": selected, "all_candidates": all_candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = evaluate(selected, test, out_dir)
    summary.to_csv(out_dir / "threshold_sweep_summary.csv", index=False)

    table = summary.copy()
    if args.ransac_summary:
        ransac_path = Path(args.ransac_summary)
        if not ransac_path.is_file():
            raise FileNotFoundError(ransac_path)
        ransac = pd.read_csv(ransac_path)
        ransac = ransac[ransac["method"] == "ransac_multiplane_residual"].copy()
        common = [column for column in table.columns if column in ransac.columns]
        table = pd.concat([table, ransac[common]], ignore_index=True, sort=False)
    table.to_csv(out_dir / "math_baselines_table.csv", index=False)
    print("\n[TEST: thresholds selected on validation]\n" + summary.to_string(index=False))
    print(f"\n[DONE] {out_dir / 'threshold_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
