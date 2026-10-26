#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def metrics_from_arrays(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou_pos = tp / max(tp + fp + fn, 1)
    iou_neg = tn / max(tn + fp + fn, 1)
    return {
        "miou": float((iou_pos + iou_neg) / 2.0),
        "acc": float(acc),
        "recall": float(recall),
        "f1": float(f1),
    }


def pred_to_rgb(pred):
    pred = np.asarray(pred).astype(np.int32).reshape(-1)
    rgb = np.full((pred.size, 3), 160, dtype=np.uint8)
    rgb[pred == 1] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[pred < 0] = np.array([60, 60, 60], dtype=np.uint8)
    return rgb


def prob_to_gray_rgb(prob):
    prob = np.clip(np.asarray(prob, dtype=np.float32).reshape(-1), 0.0, 1.0)
    gray = (255.0 * (1.0 - prob)).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=1)


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def read_prediction_file(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        xyz = df[["x", "y", "z"]].to_numpy() if all(c in df.columns for c in ["x", "y", "z"]) else None
        pred_col = "pred" if "pred" in df.columns else "prediction" if "prediction" in df.columns else None
        label_col = "label" if "label" in df.columns else "gt" if "gt" in df.columns else "scalar_label" if "scalar_label" in df.columns else None
        prob = df["prob"].to_numpy() if "prob" in df.columns else None
        pred = df[pred_col].to_numpy() if pred_col else None
        label = df[label_col].to_numpy() if label_col else None
        return xyz, label, pred, prob
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    xyz = arr[:, :3]
    label = arr[:, 3].astype(np.int32) if arr.shape[1] >= 5 else None
    pred = arr[:, 4].astype(np.int32) if arr.shape[1] >= 5 else arr[:, 3].astype(np.int32)
    return xyz, label, pred, None


def parse_log_metrics(log_path):
    if not log_path or not os.path.exists(log_path):
        return {}
    text = Path(log_path).read_text(errors="ignore")
    patterns = {
        "miou": r"(?:miou|mIoU)[:=]\s*([0-9.]+)",
        "acc": r"(?:oa|accuracy|acc)[:=]\s*([0-9.]+)",
        "recall": r"(?:weld_recall|recall)[:=]\s*([0-9.]+)",
        "f1": r"(?:weld_f1|f1|F1)[:=]\s*([0-9.]+)",
    }
    out = {}
    for k, pat in patterns.items():
        vals = re.findall(pat, text, flags=re.IGNORECASE)
        if vals:
            out[k] = float(vals[-1])
    best_epoch = None
    best_val = -1.0
    for m in re.finditer(r"\[Epoch\s+(\d+)/(?:\d+)\].*?(?:val_miou|mIoU)[:=]([0-9.]+)", text):
        val = float(m.group(2))
        if val > best_val:
            best_val = val
            best_epoch = int(m.group(1))
    if best_epoch is not None:
        out["best_epoch"] = best_epoch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_points", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--model_path", default="")
    ap.add_argument("--pred_dir", required=True, help="Directory containing fullpred csv or xyz-label txt files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--log_path", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pc_dir = out_dir / "pointclouds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in Path(args.pred_dir).glob("*") if p.suffix.lower() in [".csv", ".txt"]])
    labels_all, preds_all = [], []
    rows = []
    for p in files:
        xyz, label, pred, prob = read_prediction_file(p)
        if xyz is not None and pred is not None:
            stem = p.stem.replace("_fullpred", "").replace("_pred_xyz_label", "")
            write_ply(pc_dir / f"{stem}_pred.ply", xyz, pred_to_rgb(pred))
            if prob is not None:
                write_ply(pc_dir / f"{stem}_prob.ply", xyz, prob_to_gray_rgb(prob))
        if label is not None and pred is not None:
            m = metrics_from_arrays(label, pred)
            rows.append({"file": p.name, **m})
            labels_all.append(label)
            preds_all.append(pred)

    log_metrics = parse_log_metrics(args.log_path)
    if labels_all:
        summary = metrics_from_arrays(np.concatenate(labels_all), np.concatenate(preds_all))
    else:
        summary = {k: float(log_metrics.get(k, 0.0)) for k in ["miou", "acc", "recall", "f1"]}

    summary.update({
        "method": args.method,
        "exp_name": args.exp_name,
        "seed": args.seed,
        "max_points": args.max_points,
        "epochs": args.epochs,
        "best_epoch": log_metrics.get("best_epoch", ""),
        "model_path": args.model_path,
        "pred_dir": args.pred_dir,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    pd.DataFrame([summary]).to_csv(out_dir / "eval_summary_metrics.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "per_file_metrics.csv", index=False)

    report = f"""# 实验记录

## 基本信息
- 实验名称：`{args.exp_name}`
- 对比方法：`{args.method}`
- 随机种子：`{args.seed}`
- 最大点数：`{args.max_points}`
- 运行时间：`{summary['created_at']}`

## 模型与训练配置
- 模型路径：`{args.model_path}`
- 训练 epoch：`{args.epochs}`
- 最好 epoch：`{summary['best_epoch']}`
- 预测目录：`{args.pred_dir}`

## 推理结果
- mIoU：`{summary['miou']:.6f}`
- Accuracy：`{summary['acc']:.6f}`
- Recall：`{summary['recall']:.6f}`
- F1：`{summary['f1']:.6f}`

## 输出文件
- 指标CSV：`eval_summary_metrics.csv`
- 逐文件指标：`per_file_metrics.csv`
- 点云文件：`pointclouds/`
"""
    (out_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
