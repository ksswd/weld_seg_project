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


def calc_baseline_metrics(labels: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    labels = labels.astype(np.int32).reshape(-1)
    preds = (probs > thr).astype(np.int32).reshape(-1)

    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    oa = (tp + tn) / max(tp + tn + fp + fn, 1)
    class_iou_0 = tn / max(tn + fp + fn, 1)
    class_iou_1 = tp / max(tp + fp + fn, 1)
    class_acc_0 = tn / max(tn + fp, 1)
    class_acc_1 = tp / max(tp + fn, 1)

    miou = (class_iou_0 + class_iou_1) / 2.0
    macc = (class_acc_0 + class_acc_1) / 2.0

    weld_precision = tp / max(tp + fp, 1)
    weld_recall = tp / max(tp + fn, 1)
    weld_f1 = 2 * weld_precision * weld_recall / max(weld_precision + weld_recall, 1e-12)
    weld_iou = class_iou_1

    return {
        "oa": float(oa),
        "macc": float(macc),
        "miou": float(miou),
        "class_iou_0": float(class_iou_0),
        "class_iou_1": float(class_iou_1),
        "class_acc_0": float(class_acc_0),
        "class_acc_1": float(class_acc_1),
        "weld_precision": float(weld_precision),
        "weld_recall": float(weld_recall),
        "weld_f1": float(weld_f1),
        "weld_iou": float(weld_iou),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
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


def _eval_one_target(labels_bin: np.ndarray, probs: np.ndarray, fixed_thr: float, n_steps: int, prefix: str) -> Dict[str, float]:
    pred_fixed = (probs > fixed_thr).astype(np.int32)
    conf_fixed = calc_confusion(labels_bin, pred_fixed)
    m_fixed = calc_metrics_from_conf(*conf_fixed)

    t_best, m_best, conf_best = best_threshold(probs, labels_bin, n_steps=n_steps)

    return {
        f"{prefix}_pos_ratio": float(labels_bin.mean()),
        f"{prefix}_best_thr": float(t_best),

        f"{prefix}_acc@fixed": m_fixed["accuracy"],
        f"{prefix}_prec@fixed": m_fixed["precision"],
        f"{prefix}_rec@fixed": m_fixed["recall"],
        f"{prefix}_f1@fixed": m_fixed["f1"],
        f"{prefix}_iou@fixed": m_fixed["iou"],
        f"{prefix}_spec@fixed": m_fixed["specificity"],

        f"{prefix}_acc@best": m_best["accuracy"],
        f"{prefix}_prec@best": m_best["precision"],
        f"{prefix}_rec@best": m_best["recall"],
        f"{prefix}_f1@best": m_best["f1"],
        f"{prefix}_iou@best": m_best["iou"],
        f"{prefix}_spec@best": m_best["specificity"],

        f"{prefix}_tp@best": int(conf_best[0]),
        f"{prefix}_tn@best": int(conf_best[1]),
        f"{prefix}_fp@best": int(conf_best[2]),
        f"{prefix}_fn@best": int(conf_best[3]),
    }


def process_one_file(file_path: str, fixed_thr: float, n_steps: int, soft_thr: float) -> Dict[str, float]:
    df = pd.read_csv(file_path)
    required = {"label", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path} 缺少列: {missing}")

    probs = df["prob"].to_numpy(dtype=np.float32)

    # hard 口径
    labels_hard = (df["label"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)

    # soft-aware 口径（若无label_soft则回退到hard）
    if "label_soft" in df.columns:
        labels_soft = (df["label_soft"].to_numpy(dtype=np.float32) > float(soft_thr)).astype(np.int32)
    else:
        labels_soft = labels_hard

    row = {
        "file": os.path.basename(file_path),
        "n_points": int(len(labels_hard)),
        "fixed_thr": float(fixed_thr),
        "soft_thr": float(soft_thr),
    }
    row.update(_eval_one_target(labels_hard, probs, fixed_thr, n_steps, prefix="hard"))
    row.update(_eval_one_target(labels_soft, probs, fixed_thr, n_steps, prefix="soft"))

    baseline_hard = calc_baseline_metrics(labels_hard, probs, fixed_thr)
    baseline_soft = calc_baseline_metrics(labels_soft, probs, fixed_thr)
    row.update({f"hard_{k}": v for k, v in baseline_hard.items()})
    row.update({f"soft_{k}": v for k, v in baseline_soft.items()})

    return row


def main():
    ap = argparse.ArgumentParser(description="计算 full_eval 每样例量化指标")
    ap.add_argument("--input_dir", required=True, help="包含 *_fullpred.csv 的目录")
    ap.add_argument("--pattern", default="*_fullpred.csv", help="文件匹配模式")
    ap.add_argument("--fixed_thr", type=float, default=0.53, help="固定阈值（用于统一横向比较）")
    ap.add_argument("--n_steps", type=int, default=201, help="搜索最佳阈值的步数")
    ap.add_argument("--soft_thr", type=float, default=0.5, help="soft-aware标签阈值（label_soft > soft_thr 视为正类）")
    ap.add_argument("--out_per_case", default="summary_metrics_per_case.csv", help="每样例输出文件名")
    ap.add_argument("--out_global", default="summary_metrics_global.csv", help="全局汇总输出文件名")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"未找到文件: {args.input_dir}/{args.pattern}")

    rows = []
    all_probs = []
    all_labels_hard = []
    all_labels_soft = []

    for f in files:
        row = process_one_file(f, fixed_thr=args.fixed_thr, n_steps=args.n_steps, soft_thr=args.soft_thr)
        rows.append(row)

        d = pd.read_csv(f)
        y_hard = (d["label"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        if "label_soft" in d.columns:
            y_soft = (d["label_soft"].to_numpy(dtype=np.float32) > float(args.soft_thr)).astype(np.int32)
        else:
            y_soft = y_hard
        p = d["prob"].to_numpy(dtype=np.float32)
        all_labels_hard.append(y_hard)
        all_labels_soft.append(y_soft)
        all_probs.append(p)

    df_case = pd.DataFrame(rows).sort_values("hard_f1@best", ascending=False)
    out_case = os.path.join(args.input_dir, args.out_per_case)
    df_case.to_csv(out_case, index=False)

    # 全局（聚合所有点）：hard + soft 双口径
    labels_hard_all = np.concatenate(all_labels_hard)
    labels_soft_all = np.concatenate(all_labels_soft)
    probs_all = np.concatenate(all_probs)

    hard_global = _eval_one_target(labels_hard_all, probs_all, fixed_thr=args.fixed_thr, n_steps=args.n_steps, prefix="hard")
    soft_global = _eval_one_target(labels_soft_all, probs_all, fixed_thr=args.fixed_thr, n_steps=args.n_steps, prefix="soft")

    baseline_hard_global = calc_baseline_metrics(labels_hard_all, probs_all, args.fixed_thr)
    baseline_soft_global = calc_baseline_metrics(labels_soft_all, probs_all, args.fixed_thr)

    df_global = pd.DataFrame([
        {
            "n_files": len(files),
            "n_points": int(len(labels_hard_all)),
            "fixed_thr": float(args.fixed_thr),
            "soft_thr": float(args.soft_thr),
            **hard_global,
            **soft_global,
            **{f"hard_{k}": v for k, v in baseline_hard_global.items()},
            **{f"soft_{k}": v for k, v in baseline_soft_global.items()},
        }
    ])
    out_global = os.path.join(args.input_dir, args.out_global)
    df_global.to_csv(out_global, index=False)

    print(f"[done] per-case metrics -> {out_case}")
    print(f"[done] global metrics   -> {out_global}")

    def _print_baseline(prefix: str, metrics: Dict[str, float]):
        print(f"\n=== {prefix} baseline metrics (fixed_thr={args.fixed_thr}) ===")
        print(f"oa: {metrics['oa']:.6f}")
        print(f"macc: {metrics['macc']:.6f}")
        print(f"miou: {metrics['miou']:.6f}")
        print(f"class_iou: [{metrics['class_iou_0']:.6f}, {metrics['class_iou_1']:.6f}]")
        print(f"class_acc: [{metrics['class_acc_0']:.6f}, {metrics['class_acc_1']:.6f}]")
        print(f"weld_precision: {metrics['weld_precision']:.6f}")
        print(f"weld_recall: {metrics['weld_recall']:.6f}")
        print(f"weld_f1: {metrics['weld_f1']:.6f}")
        print(f"weld_iou: {metrics['weld_iou']:.6f}")

    _print_baseline("hard", baseline_hard_global)


if __name__ == "__main__":
    main()
