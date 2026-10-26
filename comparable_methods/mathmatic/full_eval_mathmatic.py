import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FileEvalResult:
    file: str
    n_points: int
    pos_ratio: float
    pred_ratio: float
    f1: float
    precision: float
    recall: float
    iou: float
    accuracy: float


def calc_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def calc_metrics(tp: int, tn: int, fp: int, fn: int) -> Tuple[float, float, float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return f1, precision, recall, iou, acc


def load_pred_from_df(df: pd.DataFrame, pred_col: str, threshold: float) -> np.ndarray:
    if pred_col in df.columns:
        pred = df[pred_col].to_numpy(dtype=np.float32)
        uniq = np.unique(pred)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            return pred.astype(np.int32)
        return (pred > threshold).astype(np.int32)
    if "prob" in df.columns:
        return (df["prob"].to_numpy(dtype=np.float32) > threshold).astype(np.int32)
    raise ValueError(f"未找到预测列: {pred_col} 或 prob")


def eval_one_file(file_path: str, pred_col: str, threshold: float) -> Tuple[FileEvalResult, Tuple[int, int, int, int]]:
    df = pd.read_csv(file_path)
    if "label" not in df.columns:
        raise ValueError(f"{file_path} 缺少列: label")

    labels = (df["label"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
    preds = load_pred_from_df(df, pred_col=pred_col, threshold=threshold)

    tp, tn, fp, fn = calc_confusion(labels, preds)
    f1, precision, recall, iou, acc = calc_metrics(tp, tn, fp, fn)

    result = FileEvalResult(
        file=os.path.basename(file_path),
        n_points=int(labels.shape[0]),
        pos_ratio=float(labels.mean()),
        pred_ratio=float(preds.mean()),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        iou=float(iou),
        accuracy=float(acc),
    )
    return result, (tp, tn, fp, fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估数学方法输出的逐点分割指标")
    parser.add_argument("--input_dir", required=True, help="包含 *_pred_method*.csv 的目录")
    parser.add_argument("--pattern", default="*_pred_method*.csv", help="文件匹配模式")
    parser.add_argument("--pred_col", default="pred", help="预测列名，默认pred；若无则尝试prob")
    parser.add_argument("--threshold", type=float, default=0.5, help="当预测列为连续值时的阈值")
    parser.add_argument("--out_per_file", default="summary.csv", help="逐文件结果名")
    parser.add_argument("--out_global", default="summary_global.csv", help="全局聚合结果名")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"未找到文件: {args.input_dir}/{args.pattern}")

    results: List[FileEvalResult] = []
    total_tp = total_tn = total_fp = total_fn = 0

    for f in files:
        res, (tp, tn, fp, fn) = eval_one_file(f, pred_col=args.pred_col, threshold=args.threshold)
        results.append(res)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        print(
            f"[{res.file}] N={res.n_points} pos={res.pos_ratio:.4f} pred={res.pred_ratio:.4f} "
            f"F1={res.f1:.4f} P={res.precision:.4f} R={res.recall:.4f} IoU={res.iou:.4f} Acc={res.accuracy:.4f}"
        )

    df_sum = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1", ascending=False)
    out_per = os.path.join(args.input_dir, args.out_per_file)
    df_sum.to_csv(out_per, index=False)

    g_f1, g_p, g_r, g_iou, g_acc = calc_metrics(total_tp, total_tn, total_fp, total_fn)
    global_row = {
        "n_files": len(results),
        "n_points": int(df_sum["n_points"].sum()),
        "global_f1": float(g_f1),
        "global_precision": float(g_p),
        "global_recall": float(g_r),
        "global_iou": float(g_iou),
        "global_accuracy": float(g_acc),
        "tp": int(total_tp),
        "tn": int(total_tn),
        "fp": int(total_fp),
        "fn": int(total_fn),
    }
    out_global = os.path.join(args.input_dir, args.out_global)
    pd.DataFrame([global_row]).to_csv(out_global, index=False)

    print("\n=== SUMMARY (aggregated) ===")
    print(f"Global F1       : {g_f1:.4f}")
    print(f"Global Precision: {g_p:.4f}")
    print(f"Global Recall   : {g_r:.4f}")
    print(f"Global IoU      : {g_iou:.4f}")
    print(f"Global Accuracy : {g_acc:.4f}")
    print(f"Wrote per-file summary: {out_per}")
    print(f"Wrote global summary  : {out_global}")


if __name__ == "__main__":
    main()
