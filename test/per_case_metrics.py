#!/usr/bin/env python3
"""
基于 full_eval 输出的 *_fullpred.csv 计算“每个样例”的量化指标，便于复用。

输入要求：
- 每个 *_fullpred.csv 至少包含列: label, prob

输出：
- summary_metrics_per_case.csv（每样例指标）
- summary_metrics_global.csv（全局汇总）

用法示例：
python test/per_case_metrics.py --input_dir data/new/full_eval_all --fixed_thr 0.53
"""

import argparse
import glob
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def calc_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def calc_metrics_from_conf(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "specificity": float(spec),
    }


def best_threshold(probs: np.ndarray, labels: np.ndarray, n_steps: int = 201) -> Tuple[float, Dict[str, float], Tuple[int, int, int, int]]:
    best_f1 = -1.0
    best_t = 0.5
    best_metrics = None
    best_conf = None

    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (probs > t).astype(np.int32)
        conf = calc_confusion(labels, pred)
        m = calc_metrics_from_conf(*conf)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
            best_conf = conf

    return best_t, best_metrics, best_conf


def process_one_file(file_path: str, fixed_thr: float, n_steps: int) -> Dict[str, float]:
    df = pd.read_csv(file_path)
    required = {"label", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} 缺少列: {missing}")

    labels = df["label"].to_numpy(dtype=np.float32)
    labels = (labels > 0.5).astype(np.int32)
    probs = df["prob"].to_numpy(dtype=np.float32)

    # 固定阈值指标
    pred_fixed = (probs > fixed_thr).astype(np.int32)
    conf_fixed = calc_confusion(labels, pred_fixed)
    m_fixed = calc_metrics_from_conf(*conf_fixed)

    # 最佳阈值指标
    t_best, m_best, conf_best = best_threshold(probs, labels, n_steps=n_steps)

    row = {
        "file": os.path.basename(file_path),
        "n_points": int(len(labels)),
        "pos_ratio": float(labels.mean()),
        "fixed_thr": float(fixed_thr),
        "best_thr": float(t_best),

        "acc@fixed": m_fixed["accuracy"],
        "prec@fixed": m_fixed["precision"],
        "rec@fixed": m_fixed["recall"],
        "f1@fixed": m_fixed["f1"],
        "iou@fixed": m_fixed["iou"],
        "spec@fixed": m_fixed["specificity"],

        "acc@best": m_best["accuracy"],
        "prec@best": m_best["precision"],
        "rec@best": m_best["recall"],
        "f1@best": m_best["f1"],
        "iou@best": m_best["iou"],
        "spec@best": m_best["specificity"],

        "tp@best": int(conf_best[0]),
        "tn@best": int(conf_best[1]),
        "fp@best": int(conf_best[2]),
        "fn@best": int(conf_best[3]),
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="计算 full_eval 每样例量化指标")
    ap.add_argument("--input_dir", required=True, help="包含 *_fullpred.csv 的目录")
    ap.add_argument("--pattern", default="*_fullpred.csv", help="文件匹配模式")
    ap.add_argument("--fixed_thr", type=float, default=0.53, help="固定阈值（用于统一横向比较）")
    ap.add_argument("--n_steps", type=int, default=201, help="搜索最佳阈值的步数")
    ap.add_argument("--out_per_case", default="summary_metrics_per_case.csv", help="每样例输出文件名")
    ap.add_argument("--out_global", default="summary_metrics_global.csv", help="全局汇总输出文件名")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"未找到文件: {args.input_dir}/{args.pattern}")

    rows = []
    all_probs = []
    all_labels = []

    for f in files:
        row = process_one_file(f, fixed_thr=args.fixed_thr, n_steps=args.n_steps)
        rows.append(row)

        d = pd.read_csv(f, usecols=["label", "prob"])
        y = (d["label"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        p = d["prob"].to_numpy(dtype=np.float32)
        all_labels.append(y)
        all_probs.append(p)

    df_case = pd.DataFrame(rows).sort_values("f1@best", ascending=False)
    out_case = os.path.join(args.input_dir, args.out_per_case)
    df_case.to_csv(out_case, index=False)

    # 全局（聚合所有点）
    labels_all = np.concatenate(all_labels)
    probs_all = np.concatenate(all_probs)

    pred_fixed_all = (probs_all > args.fixed_thr).astype(np.int32)
    conf_fixed_all = calc_confusion(labels_all, pred_fixed_all)
    m_fixed_all = calc_metrics_from_conf(*conf_fixed_all)

    t_best_all, m_best_all, conf_best_all = best_threshold(probs_all, labels_all, n_steps=args.n_steps)

    df_global = pd.DataFrame([
        {
            "n_files": len(files),
            "n_points": int(len(labels_all)),
            "pos_ratio": float(labels_all.mean()),
            "fixed_thr": float(args.fixed_thr),
            "best_thr": float(t_best_all),

            "acc@fixed": m_fixed_all["accuracy"],
            "prec@fixed": m_fixed_all["precision"],
            "rec@fixed": m_fixed_all["recall"],
            "f1@fixed": m_fixed_all["f1"],
            "iou@fixed": m_fixed_all["iou"],
            "spec@fixed": m_fixed_all["specificity"],

            "acc@best": m_best_all["accuracy"],
            "prec@best": m_best_all["precision"],
            "rec@best": m_best_all["recall"],
            "f1@best": m_best_all["f1"],
            "iou@best": m_best_all["iou"],
            "spec@best": m_best_all["specificity"],

            "tp@best": int(conf_best_all[0]),
            "tn@best": int(conf_best_all[1]),
            "fp@best": int(conf_best_all[2]),
            "fn@best": int(conf_best_all[3]),
        }
    ])
    out_global = os.path.join(args.input_dir, args.out_global)
    df_global.to_csv(out_global, index=False)

    print(f"[done] per-case metrics -> {out_case}")
    print(f"[done] global metrics   -> {out_global}")


if __name__ == "__main__":
    main()
