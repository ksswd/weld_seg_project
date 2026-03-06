#!/usr/bin/env python3
"""
为每个样例绘制指标柱状图（共19个样例可放在一张总图中）。

指标定义：
1) 准确率（Precision）: TP / (TP + FP)
2) 完备率（Recall）: TP / (TP + FN)
3) 错误率（ErrorRate）: (FP + FN) / N
4) 误差累积量（ErrDistSum）:
   对每个错误点(y_pred != y_true)，计算其到“最近正确点(y_pred == y_true)”的欧式距离并求和。

图中每个子图标题下方会显示：
- 总点数N
- 预测点数量(PredPos=TP+FP)
- 正确点数量(正确预测焊缝点，TP)
- 错误点数量(错误预测焊缝点，FP)
"""

import argparse
import glob
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def calc_error_distance_sum(xyz: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err_mask = (y_true != y_pred)
    ok_mask = ~err_mask

    if err_mask.sum() == 0:
        return 0.0
    if ok_mask.sum() == 0:
        return float("nan")

    tree = cKDTree(xyz[ok_mask])
    dists, _ = tree.query(xyz[err_mask], k=1, workers=-1)
    return float(np.sum(dists))


def compute_case_metrics(df: pd.DataFrame, thr: float) -> Dict[str, float]:
    required = {"x", "y", "z", "label", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入CSV缺少列: {missing}")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    y_true = (df["label"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
    y_pred = (df["prob"].to_numpy(dtype=np.float32) > thr).astype(np.int32)

    tp, tn, fp, fn = confusion(y_true, y_pred)
    n = len(y_true)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    error_rate = (fp + fn) / max(n, 1)
    err_dist_sum = calc_error_distance_sum(xyz, y_true, y_pred)

    return {
        "n_points": int(n),
        "pred_pos": int(tp + fp),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "error_rate": float(error_rate),
        "err_dist_sum": float(err_dist_sum),
    }


def plot_all_cases(case_rows: pd.DataFrame, out_png: str):
    n = len(case_rows)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), dpi=160)
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(case_rows.iterrows()):
        ax = axes[i]

        # 前三项统一转百分比(0-100)，最后一项保持mm
        vals = [
            row["precision"] * 100.0,
            row["recall"] * 100.0,
            row["error_rate"] * 100.0,
            row["err_dist_sum"],
        ]
        names = ["Precision(%)", "Recall(%)", "ErrorRate(%)", "ErrDistSum(mm)"]
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

        bars = ax.bar(names, vals, color=colors, alpha=0.9)
        ax.set_xticklabels(names, rotation=25, ha="right")

        for bi, (b, v) in enumerate(zip(bars, vals)):
            if np.isnan(v):
                txt = "NaN"
            else:
                # 前三项显示百分比，最后一项显示mm
                if bi < 3:
                    txt = f"{v:.1f}%"
                else:
                    txt = f"{v:.1f} mm"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                    ha="center", va="bottom", fontsize=8)

        name = row["file"].replace("_fullpred.csv", "")
        ax.set_title(
            f"{name}\n"
            f"N={int(row['n_points'])} | PredPos={int(row['pred_pos'])} | TP={int(row['tp'])} | FP={int(row['fp'])}",
            fontsize=10
        )
        # 强制统一纵轴范围到 0-100，便于跨样例直观比较
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-Case Metrics Overview", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="绘制每个样例的指标柱状图")
    ap.add_argument("--input_dir", required=True, help="包含 *_fullpred.csv 的目录")
    ap.add_argument("--pattern", default="*_fullpred.csv", help="匹配模式")
    ap.add_argument("--thr", type=float, default=0.53, help="二值化阈值")
    ap.add_argument("--out_png", default="case_metrics_bars.png", help="输出总图文件名")
    ap.add_argument("--out_csv", default="case_metrics_custom.csv", help="输出指标CSV文件名")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"未找到文件: {args.input_dir}/{args.pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        m = compute_case_metrics(df, thr=args.thr)
        m["file"] = os.path.basename(f)
        rows.append(m)

    case_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)

    out_csv = os.path.join(args.input_dir, args.out_csv)
    case_df.to_csv(out_csv, index=False)

    out_png = os.path.join(args.input_dir, args.out_png)
    plot_all_cases(case_df, out_png)

    print(f"[done] metrics csv: {out_csv}")
    print(f"[done] plot image : {out_png}")


if __name__ == "__main__":
    main()
