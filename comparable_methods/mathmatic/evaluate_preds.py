#!/usr/bin/env python3
"""
对数学方法预测结果做逐文件指标统计，并生成汇总与柱状图。

输入：包含 *_pred_method1.csv / *_pred_method2.csv 的目录
要求列：x,y,z,label,pred
输出：
- metrics_per_case.csv
- metrics_summary.csv
- metrics_bars.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_metrics(label: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="预测结果目录")
    parser.add_argument("--out", default=None, help="输出目录，默认与input_dir相同")
    parser.add_argument("--plot_topk", type=int, default=30, help="柱状图最多展示前k个样本")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out or input_dir
    os.makedirs(out_dir, exist_ok=True)

    files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if f.endswith("_pred_method1.csv") or f.endswith("_pred_method2.csv")
    ]
    if not files:
        raise SystemExit(f"未找到预测文件: {input_dir}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        required = {"label", "pred"}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"{f} 缺少列: {miss}")

        label = df["label"].to_numpy(dtype=np.int32)
        pred = df["pred"].to_numpy(dtype=np.int32)
        m = compute_metrics(label, pred)
        m["file"] = Path(f).name
        m["points"] = len(df)
        rows.append(m)

    per_case = pd.DataFrame(rows)[
        ["file", "points", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "acc"]
    ]
    per_case.to_csv(os.path.join(out_dir, "metrics_per_case.csv"), index=False)

    summary = {
        "num_files": len(per_case),
        "precision_mean": per_case["precision"].mean(),
        "precision_std": per_case["precision"].std(ddof=0),
        "recall_mean": per_case["recall"].mean(),
        "recall_std": per_case["recall"].std(ddof=0),
        "f1_mean": per_case["f1"].mean(),
        "f1_std": per_case["f1"].std(ddof=0),
        "acc_mean": per_case["acc"].mean(),
        "acc_std": per_case["acc"].std(ddof=0),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # 画图（按f1降序）
    show_df = per_case.sort_values("f1", ascending=False).head(args.plot_topk)
    x = np.arange(len(show_df))
    width = 0.25

    plt.figure(figsize=(max(10, len(show_df) * 0.35), 5), dpi=150)
    plt.bar(x - width, show_df["precision"].values, width=width, label="precision")
    plt.bar(x, show_df["recall"].values, width=width, label="recall")
    plt.bar(x + width, show_df["f1"].values, width=width, label="f1")
    plt.ylim(0, 1.0)
    plt.xticks(x, [Path(n).stem for n in show_df["file"]], rotation=70, ha="right", fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_bars.png"))
    plt.close()

    print(f"[saved] {os.path.join(out_dir, 'metrics_per_case.csv')}")
    print(f"[saved] {os.path.join(out_dir, 'metrics_summary.csv')}")
    print(f"[saved] {os.path.join(out_dir, 'metrics_bars.png')}")


if __name__ == "__main__":
    main()
