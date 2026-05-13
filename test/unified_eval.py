import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-8


def f1_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def best_threshold_f1(probs: np.ndarray, labels: np.ndarray, n_steps: int = 201) -> Tuple[float, float]:
    best_f1 = 0.0
    best_t = 0.5
    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (probs > t).astype(np.int32)
        f1 = f1_score_numpy(labels, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_f1, best_t


def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)

    conf = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(target, pred):
        if 0 <= t < 2 and 0 <= p < 2:
            conf[t, p] += 1

    oa = np.trace(conf) / float(conf.sum() + EPS)

    class_iou = []
    class_acc = []
    for c in range(2):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp
        iou = tp / float(tp + fp + fn + EPS)
        acc = tp / float(conf[c, :].sum() + EPS)
        class_iou.append(float(iou))
        class_acc.append(float(acc))

    miou = float(np.mean(class_iou))
    macc = float(np.mean(class_acc))

    tp = conf[1, 1]
    fn = conf[1, :].sum() - tp
    fp = conf[:, 1].sum() - tp
    precision = tp / float(tp + fp + EPS)
    recall = tp / float(tp + fn + EPS)
    f1 = 2 * precision * recall / float(precision + recall + EPS)

    return {
        "oa": float(oa),
        "macc": float(macc),
        "miou": float(miou),
        "class0_iou": float(class_iou[0]),
        "class1_iou": float(class_iou[1]),
        "class0_acc": float(class_acc[0]),
        "class1_acc": float(class_acc[1]),
        "weld_precision": float(precision),
        "weld_recall": float(recall),
        "weld_f1": float(f1),
        "weld_iou": float(class_iou[1]),
    }


def resolve_files(input_glob: str, input_dir: str, suffix: str) -> List[str]:
    if input_glob:
        files = sorted(glob.glob(input_glob))
    else:
        files = sorted(glob.glob(os.path.join(input_dir, f"*{suffix}")))
    return files


def to_binary_pred_from_column(arr: np.ndarray, threshold: float) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating):
        return (arr > threshold).astype(np.int32)
    return arr.astype(np.int32)


def evaluate_one_file(
    file_path: str,
    label_col: str,
    pred_col: str,
    prob_col: str,
    threshold: float,
    prefer_prob: bool,
):
    df = pd.read_csv(file_path)
    if label_col not in df.columns:
        raise ValueError(f"{file_path} missing label column: {label_col}")

    labels = df[label_col].to_numpy().astype(np.int32)
    valid_label_mask = np.isin(labels, [0, 1])

    has_pred = pred_col in df.columns
    has_prob = prob_col in df.columns

    if prefer_prob and has_prob:
        probs = df[prob_col].to_numpy().astype(np.float32)
        pred = (probs > threshold).astype(np.int32)
        pred_source = f"prob>{threshold}"
    elif has_pred:
        pred_raw = df[pred_col].to_numpy()
        pred = to_binary_pred_from_column(pred_raw, threshold)
        probs = None
        pred_source = pred_col
    elif has_prob:
        probs = df[prob_col].to_numpy().astype(np.float32)
        pred = (probs > threshold).astype(np.int32)
        pred_source = f"prob>{threshold}"
    else:
        raise ValueError(f"{file_path} missing both pred_col({pred_col}) and prob_col({prob_col})")

    valid_mask = valid_label_mask & np.isin(pred, [0, 1])
    labels_v = labels[valid_mask]
    pred_v = pred[valid_mask]

    if labels_v.size == 0:
        raise ValueError(f"{file_path} has no valid rows with binary label/pred")

    metrics = compute_binary_metrics(pred_v, labels_v)

    row = {
        "file": os.path.basename(file_path),
        "n_rows": int(df.shape[0]),
        "n_valid": int(labels_v.size),
        "pos_ratio": float((labels_v == 1).mean()),
        "pred_source": pred_source,
        "threshold": float(threshold),
        **metrics,
    }

    probs_v = None
    if has_prob:
        probs_v = df[prob_col].to_numpy().astype(np.float32)[valid_mask]
        f1_05 = f1_score_numpy(labels_v, (probs_v > 0.5).astype(np.int32))
        best_f1, best_t = best_threshold_f1(probs_v, labels_v)
        row.update({
            "f1_at_05_from_prob": float(f1_05),
            "best_f1_from_prob": float(best_f1),
            "best_thr_from_prob": float(best_t),
        })

    return row, labels_v, pred_v, probs_v


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified evaluation for weld segmentation predictions.")
    ap.add_argument("--input_glob", type=str, default="", help="Glob for prediction CSV files, e.g. data/predictions/full_eval/*_fullpred.csv")
    ap.add_argument("--input_dir", type=str, default="data/predictions/full_eval", help="Input directory when input_glob is empty")
    ap.add_argument("--suffix", type=str, default="_fullpred.csv", help="Suffix used with input_dir")

    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--pred_col", type=str, default="prediction", help="Hard prediction column (0/1 or float)")
    ap.add_argument("--prob_col", type=str, default="prob", help="Probability column")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for prob or float pred")
    ap.add_argument("--prefer_prob", action="store_true", help="Prefer prob_col even if pred_col exists")

    ap.add_argument("--out_dir", type=str, default="data/predictions/unified_eval")
    ap.add_argument("--summary_name", type=str, default="summary_unified.csv")
    ap.add_argument("--global_name", type=str, default="global_unified.csv")
    return ap


def main():
    args = build_arg_parser().parse_args()

    files = resolve_files(args.input_glob, args.input_dir, args.suffix)
    if not files:
        raise SystemExit("No input files found. Please check --input_glob/--input_dir/--suffix")

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    all_labels = []
    all_pred = []
    all_probs = []
    has_any_prob = False

    for fp in files:
        row, labels_v, pred_v, probs_v = evaluate_one_file(
            file_path=fp,
            label_col=args.label_col,
            pred_col=args.pred_col,
            prob_col=args.prob_col,
            threshold=float(args.threshold),
            prefer_prob=bool(args.prefer_prob),
        )
        rows.append(row)
        all_labels.append(labels_v)
        all_pred.append(pred_v)

        if probs_v is not None:
            has_any_prob = True
            all_probs.append(probs_v)

        print(
            f"[{os.path.basename(fp)}] n_valid={row['n_valid']} "
            f"oa={row['oa']:.4f} miou={row['miou']:.4f} weld_iou={row['weld_iou']:.4f} weld_f1={row['weld_f1']:.4f}"
        )

    all_labels_np = np.concatenate(all_labels, axis=0)
    all_pred_np = np.concatenate(all_pred, axis=0)
    global_metrics = compute_binary_metrics(all_pred_np, all_labels_np)

    global_row = {
        "n_files": int(len(files)),
        "n_points": int(all_labels_np.size),
        "pos_ratio": float((all_labels_np == 1).mean()),
        "threshold": float(args.threshold),
        **global_metrics,
    }

    if has_any_prob and len(all_probs) == len(files):
        all_probs_np = np.concatenate(all_probs, axis=0)
        global_row["f1_at_05_from_prob"] = float(f1_score_numpy(all_labels_np, (all_probs_np > 0.5).astype(np.int32)))
        best_f1, best_t = best_threshold_f1(all_probs_np, all_labels_np)
        global_row["best_f1_from_prob"] = float(best_f1)
        global_row["best_thr_from_prob"] = float(best_t)

    df_summary = pd.DataFrame(rows).sort_values("weld_iou", ascending=False)
    df_global = pd.DataFrame([global_row])

    summary_path = os.path.join(args.out_dir, args.summary_name)
    global_path = os.path.join(args.out_dir, args.global_name)

    df_summary.to_csv(summary_path, index=False)
    df_global.to_csv(global_path, index=False)

    print("\n=== GLOBAL SUMMARY ===")
    print(f"n_files: {global_row['n_files']}")
    print(f"n_points: {global_row['n_points']}")
    print(f"pos_ratio: {global_row['pos_ratio']:.4f}")
    print(f"oa: {global_row['oa']:.6f}")
    print(f"macc: {global_row['macc']:.6f}")
    print(f"miou: {global_row['miou']:.6f}")
    print(f"class_iou: [{global_row['class0_iou']:.6f}, {global_row['class1_iou']:.6f}]")
    print(f"class_acc: [{global_row['class0_acc']:.6f}, {global_row['class1_acc']:.6f}]")
    print(f"weld_precision: {global_row['weld_precision']:.6f}")
    print(f"weld_recall: {global_row['weld_recall']:.6f}")
    print(f"weld_f1: {global_row['weld_f1']:.6f}")
    print(f"weld_iou: {global_row['weld_iou']:.6f}")

    if "best_f1_from_prob" in global_row:
        print(f"f1@0.5_from_prob: {global_row['f1_at_05_from_prob']:.6f}")
        print(f"best_f1_from_prob: {global_row['best_f1_from_prob']:.6f} @ thr={global_row['best_thr_from_prob']:.3f}")

    print(f"\nSaved per-file summary to: {summary_path}")
    print(f"Saved global summary to: {global_path}")


if __name__ == "__main__":
    main()
