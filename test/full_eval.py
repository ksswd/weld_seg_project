import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.model import GeometryAwareTransformer
from utils.config import Config
from utils.io_utils import load_features_from_csv


@dataclass
class FileEvalResult:
    file: str
    n_points: int
    pos_ratio: float
    covered_ratio: float
    acc_at_05: float
    miou_at_05: float
    best_miou: float
    best_thr: float
    f1_at_05: float
    recall_at_05: float
    best_f1: float
    recall_at_best_miou: float


class WeldDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        sample = load_features_from_csv(file_path)
        sample["file_path"] = file_path
        return sample


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def f1_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, fn, _ = _confusion(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def recall_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, _, fn, _ = _confusion(y_true, y_pred)
    denom = tp + fn
    return (tp / denom) if denom > 0 else 0.0


def accuracy_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    return float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0


def miou_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, fn, tn = _confusion(y_true, y_pred)
    iou_pos = tp / max(tp + fp + fn, 1)
    iou_neg = tn / max(tn + fp + fn, 1)
    return float((iou_pos + iou_neg) / 2.0)


def best_threshold_f1(probs: np.ndarray, labels: np.ndarray, n_steps: int = 201) -> Tuple[float, float]:
    best_f1 = 0.0
    best_t = 0.5
    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (probs > t).astype(np.int32)
        f1 = f1_score_numpy(labels, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_f1, best_t


def best_threshold_miou(probs: np.ndarray, labels: np.ndarray, n_steps: int = 201) -> Tuple[float, float]:
    best_miou = 0.0
    best_t = 0.5
    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (probs > t).astype(np.int32)
        miou = miou_score_numpy(labels, pred)
        if miou > best_miou:
            best_miou = miou
            best_t = float(t)
    return best_miou, best_t


def collate_fn_full(batch):
    sampled = []
    file_paths = []
    orig_sizes = []

    for item in batch:
        if item.get("labels", None) is None:
            raise KeyError("This CSV has no 'label' column. Full-eval expects labeled csv files.")
        n = item["coordinate"].shape[0]
        sampled.append({
            "features": item["features"],
            "coordinate": item["coordinate"],
            "normals": item["normals"],
            "principal_dir": item["principal_dir"],
            "curvature": item["curvature"],
            "local_density": item["local_density"],
            "linearity": item["linearity"],
            "labels": item["labels"],
        })
        file_paths.append(item["file_path"])
        orig_sizes.append(int(n))

    b = len(sampled)
    max_n = max(s["features"].shape[0] for s in sampled)

    def pad(arr_list, shape):
        out = np.full(shape, 0.0, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            n = arr.shape[0]
            out[i, :n] = arr
        return out

    feats = pad([s["features"] for s in sampled], (b, max_n, sampled[0]["features"].shape[1]))
    coordinate = pad([s["coordinate"] for s in sampled], (b, max_n, 3))
    normals = pad([s["normals"] for s in sampled], (b, max_n, 3))
    principal = pad([s["principal_dir"] for s in sampled], (b, max_n, 3))
    curvature = pad([s["curvature"] for s in sampled], (b, max_n, 1))
    density = pad([s["local_density"] for s in sampled], (b, max_n, 1))
    linearity = pad([s["linearity"] for s in sampled], (b, max_n, 1))
    labels = pad([s["labels"] for s in sampled], (b, max_n, 1))

    mask = torch.zeros(b, max_n, dtype=torch.bool)
    for i, s in enumerate(sampled):
        mask[i, : s["features"].shape[0]] = True

    return {
        "features": torch.from_numpy(feats),
        "coordinate": torch.from_numpy(coordinate),
        "normals": torch.from_numpy(normals),
        "principal_dir": torch.from_numpy(principal),
        "curvature": torch.from_numpy(curvature),
        "local_density": torch.from_numpy(density),
        "linearity": torch.from_numpy(linearity),
        "labels": torch.from_numpy(labels),
        "mask": mask,
        "file_paths": file_paths,
        "orig_sizes": orig_sizes,
    }


@torch.no_grad()
def infer_probs_chunked(
    model: GeometryAwareTransformer,
    device: torch.device,
    batch: dict,
    chunk_points: int = 1024,
    protocol_seed: int = 20260823,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    labels = batch["labels"].squeeze(-1).detach().cpu().numpy()
    mask = batch["mask"].detach().cpu().numpy()

    probs_list, labels_list = [], []
    chunk_points = max(int(chunk_points), 1)

    for bi in range(batch["features"].shape[0]):
        valid_idx = np.where(mask[bi].astype(bool))[0]
        if valid_idx.size == 0:
            probs_list.append(np.empty((0,), dtype=np.float32))
            labels_list.append(np.empty((0,), dtype=np.int32))
            continue

        file_probs = np.zeros((valid_idx.size,), dtype=np.float32)
        order = np.random.default_rng(protocol_seed).permutation(valid_idx.size)
        for start in range(0, valid_idx.size, chunk_points):
            end = min(start + chunk_points, valid_idx.size)
            out_idx = order[start:end]
            idx = valid_idx[out_idx]
            valid_n = idx.size
            if valid_n < chunk_points:
                pad = idx[np.arange(chunk_points - valid_n) % valid_n]
                idx = np.concatenate([idx, pad])
            logits = model(
                batch["features"][bi:bi + 1, idx],
                batch["coordinate"][bi:bi + 1, idx],
                batch["principal_dir"][bi:bi + 1, idx],
                batch["curvature"][bi:bi + 1, idx],
                batch["local_density"][bi:bi + 1, idx],
                batch["normals"][bi:bi + 1, idx],
                batch["linearity"][bi:bi + 1, idx],
                task="class",
            )
            chunk_prob = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().reshape(-1).astype(np.float32)
            file_probs[out_idx] = chunk_prob[:valid_n]

        probs_list.append(file_probs)
        labels_list.append(labels[bi][valid_idx].astype(np.int32))

    return probs_list, labels_list


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Full evaluation for pre-split dataset (train/validation/test).")
    ap.add_argument("--weights", default="weights/best_finetune.pth")
    ap.add_argument("--data_dir", default=Config.PROCESSED_DATA_DIR, help="Usually dataset_processed/test")
    ap.add_argument("--max_points", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol_seed", type=int, default=20260823,
                    help="fixed seed for the common full-cloud point permutation")
    ap.add_argument("--out_dir", default="data/predictions/full_eval")
    ap.add_argument("--simple_thr", type=float, default=0.5)
    return ap


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prob_to_gray_rgb(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=np.float32).reshape(-1), 0.0, 1.0)
    gray = (255.0 * (1.0 - prob)).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=1)


def pred_to_rgb(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred).reshape(-1).astype(np.int32)
    rgb = np.full((pred.size, 3), 160, dtype=np.uint8)
    rgb[pred == 1] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[pred < 0] = np.array([60, 60, 60], dtype=np.uint8)
    return rgb


def write_ascii_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _env_value(name: str, default=""):
    return os.environ.get(name, default)


def _config_value(name: str, default=""):
    return getattr(Config, name, default)


def write_experiment_report(args, eval_summary: dict, report_path: str):
    weights_dir = os.path.dirname(args.weights)
    train_summary_path = os.path.join(weights_dir, "finetune_summary.txt")
    train_summary = ""
    if os.path.exists(train_summary_path):
        with open(train_summary_path, "r", encoding="utf-8") as f:
            train_summary = f.read().strip()

    exp_name = _env_value("EXP_NAME", os.path.basename(os.path.dirname(os.path.dirname(args.out_dir))))
    lines = [
        "# 实验记录",
        "",
        "## 基本信息",
        f"- 实验名称：`{exp_name}`",
        f"- 随机种子：`{args.seed}`",
        f"- 最大采样点数：`{args.max_points}`",
        f"- 数据集路径：`{_env_value('DATA_ROOT', '')}`",
        f"- 测试数据目录：`{args.data_dir}`",
        f"- 输出目录：`{args.out_dir}`",
        f"- 运行时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## 模型配置",
        f"- 模型类型：`{_config_value('MODEL_TYPE', '')}`",
        f"- 输入维度：`{_config_value('INPUT_DIM', '')}`",
        f"- 隐藏维度：`{_config_value('D_MODEL', '')}`",
        f"- 注意力头数：`{_config_value('N_HEADS', '')}`",
        f"- 层数：`{_config_value('N_LAYERS', '')}`",
        f"- FFN维度：`{_config_value('FFN_DIM', '')}`",
        f"- 几何注意力偏置：`{_config_value('USE_GEOMETRY_BIAS', '')}`",
        f"- 边侧门控：`{_config_value('USE_SIDE_GATE', '')}`",
        f"- 曲率特征：`{_config_value('USE_CURVATURE_IN_ATTENTION', '')}`",
        f"- 密度特征：`{_config_value('USE_DENSITY_IN_ATTENTION', '')}`",
        f"- 线性度特征：`{_config_value('USE_LINEARITY_IN_ATTENTION', '')}`",
        f"- 主方向特征：`{_config_value('USE_PRINCIPAL_DIR_IN_ATTENTION', '')}`",
        f"- ALPHA0：`{_config_value('ALPHA0', '')}`",
        f"- BETA0：`{_config_value('BETA0', '')}`",
        f"- GAMMA：`{_config_value('GAMMA', '')}`",
        "",
        "## 训练配置",
        f"- 权重路径：`{args.weights}`",
        f"- 训练总 epoch：`{_config_value('NUM_EPOCHS', '')}`",
        f"- best epoch：`{_env_value('BEST_EPOCH', '见 finetune_summary.txt')}`",
        f"- simple threshold：`{args.simple_thr}`",
        "",
        "## 推理结果",
        f"- mIoU@0.5：`{eval_summary['global_miou_at_05']:.6f}`",
        f"- Accuracy@0.5：`{eval_summary['global_acc_at_05']:.6f}`",
        f"- Recall@0.5：`{eval_summary['global_recall_at_05']:.6f}`",
        f"- F1@0.5：`{eval_summary['global_f1_at_05']:.6f}`",
        f"- Best mIoU：`{eval_summary['global_best_miou']:.6f}` @ `{eval_summary['global_best_miou_thr']:.3f}`",
        "",
        "## 推理时间",
        f"- 平均单样本推理时间：`{eval_summary['avg_model_infer_time_sec_per_file']:.6f} s/file`",
        f"- 平均单点推理时间：`{eval_summary['avg_model_infer_time_ms_per_point']:.6f} ms/point`",
        f"- 总模型推理时间：`{eval_summary['total_model_infer_time_sec']:.6f} s`",
        "",
        "## 预测结果文件",
        "- full/fullpred CSV：当前 `full_eval` 目录",
        "- 预测点云 PLY：`pointclouds/`",
        "- 可视化图片：`vis/`",
    ]
    if train_summary:
        lines.extend(["", "## Finetune Summary", "", "```text", train_summary, "```"])
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_full_eval(args):
    eval_files = sorted([
                os.path.join(args.data_dir, f)
                for f in os.listdir(args.data_dir)
        if f.endswith(".csv") and "_label_" in f and "_pred" not in f
    ])
    if not eval_files:
        raise FileNotFoundError(f"No labeled csv found under {args.data_dir}")
    print(f"Eval files (pre-split): {len(eval_files)} from {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryAwareTransformer(Config).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    Config.MAX_POINTS = int(args.max_points)
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis")
    pc_dir = os.path.join(args.out_dir, "pointclouds")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    loader = DataLoader(WeldDataset(eval_files), batch_size=int(getattr(Config, "BATCH_SIZE", 1)), shuffle=False, collate_fn=collate_fn_full, pin_memory=True)

    results: List[FileEvalResult] = []
    full_probs_all, full_labels_all = [], []
    infer_time_records = []
    total_eval_start = time.perf_counter()
    # The CLI benchmark contract is authoritative.  Previously --max_points was
    # accepted by main.py but silently ignored here in favour of a 1024 default.
    chunk_points = int(getattr(args, "max_points", getattr(Config, "FULL_EVAL_CHUNK_POINTS", 1024)))

    for batch in loader:
        file_paths = batch.pop("file_paths")
        orig_sizes = batch.pop("orig_sizes")
        full_xyz_batch = batch["coordinate"].detach().cpu().numpy()

        _sync_if_cuda(device)
        infer_start = time.perf_counter()
        probs_list, labels_list = infer_probs_chunked(
            model, device, batch, chunk_points=chunk_points,
            protocol_seed=int(getattr(args, "protocol_seed", 20260823)))
        _sync_if_cuda(device)
        batch_infer_time = time.perf_counter() - infer_start
        batch_points = int(sum(len(labels) for labels in labels_list))
        batch_files = max(len(labels_list), 1)
        per_file_infer_time = batch_infer_time / batch_files
        per_point_infer_time_ms = (batch_infer_time / max(batch_points, 1)) * 1000.0

        for file_path, probs, labels, n_orig, full_xyz in zip(file_paths, probs_list, labels_list, orig_sizes, full_xyz_batch):
            infer_time_records.append({
                "file": os.path.basename(file_path),
                "full_points": int(labels.size),
                "batch_infer_time_sec": float(batch_infer_time),
                "estimated_file_infer_time_sec": float(per_file_infer_time),
                "infer_time_per_point_ms": float(per_point_infer_time_ms),
                "chunk_points": int(chunk_points),
            })
            covered = 1.0
            pred_05 = (probs > 0.5).astype(np.int32)
            acc_05 = accuracy_score_numpy(labels, pred_05)
            f1_05 = f1_score_numpy(labels, pred_05)
            recall_05 = recall_score_numpy(labels, pred_05)
            miou_05 = miou_score_numpy(labels, pred_05)
            best_miou, best_t = best_threshold_miou(probs, labels)
            pred_best_miou = (probs > best_t).astype(np.int32)
            recall_best_miou = recall_score_numpy(labels, pred_best_miou)
            best_f1, _ = best_threshold_f1(probs, labels)

            results.append(FileEvalResult(
                file=os.path.basename(file_path),
                n_points=int(labels.shape[0]),
                pos_ratio=float(labels.mean()),
                covered_ratio=float(covered),
                acc_at_05=float(acc_05),
                miou_at_05=float(miou_05),
                best_miou=float(best_miou),
                best_thr=float(best_t),
                f1_at_05=float(f1_05),
                recall_at_05=float(recall_05),
                best_f1=float(best_f1),
                recall_at_best_miou=float(recall_best_miou),
            ))

            df_full = pd.DataFrame({
                "x": full_xyz[:, 0],
                "y": full_xyz[:, 1],
                "z": full_xyz[:, 2],
                "prob": probs,
                "pred": pred_05,
                "label": labels,
            })
            out_full = os.path.join(args.out_dir, os.path.basename(file_path).replace(".csv", "_fullpred.csv"))
            df_full.to_csv(out_full, index=False)

            stem = os.path.basename(file_path).replace(".csv", "")
            write_ascii_ply(os.path.join(pc_dir, f"{stem}_prob.ply"), full_xyz, prob_to_gray_rgb(probs))
            write_ascii_ply(os.path.join(pc_dir, f"{stem}_pred.ply"), full_xyz, pred_to_rgb(pred_05))

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                full_xyz[:, 0], full_xyz[:, 1], full_xyz[:, 2],
                c=probs, s=1, cmap="viridis_r", vmin=0.0, vmax=1.0
            )
            fig.colorbar(sc, ax=ax, shrink=0.6, label="prob")
            ax.set_title(os.path.basename(file_path))
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, os.path.basename(file_path).replace(".csv", "_prob.png")), dpi=200)
            plt.close(fig)

            print(
                f"[{os.path.basename(file_path)}] fullN={labels.size} pos={labels.mean():.4f} "
                f"Acc@0.5={acc_05:.4f} mIoU@0.5={miou_05:.4f} best_mIoU={best_miou:.4f} best_miou_thr={best_t:.3f} | "
                f"F1@0.5={f1_05:.4f} Recall@0.5={recall_05:.4f} bestF1={best_f1:.4f} Recall@best_mIoU={recall_best_miou:.4f} | "
                f"infer~{per_file_infer_time:.4f}s/file, {per_point_infer_time_ms:.6f}ms/point"
            )

            full_probs_all.append(probs)
            full_labels_all.append(labels)

    total_eval_time = time.perf_counter() - total_eval_start
    full_probs_all = np.concatenate(full_probs_all)
    full_labels_all = np.concatenate(full_labels_all)
    global_pred_05 = (full_probs_all > 0.5).astype(np.int32)
    global_acc_05 = accuracy_score_numpy(full_labels_all, global_pred_05)
    global_f1_05 = f1_score_numpy(full_labels_all, global_pred_05)
    global_recall_05 = recall_score_numpy(full_labels_all, global_pred_05)
    global_best_f1, global_best_f1_t = best_threshold_f1(full_probs_all, full_labels_all)
    global_miou_05 = miou_score_numpy(full_labels_all, global_pred_05)
    global_best_miou, global_best_miou_t = best_threshold_miou(full_probs_all, full_labels_all)
    global_pred_best_miou = (full_probs_all > global_best_miou_t).astype(np.int32)
    global_recall_best_miou = recall_score_numpy(full_labels_all, global_pred_best_miou)

    print("\n=== SUMMARY (pre-split mIoU-first) ===")
    print(f"Global pos ratio      : {full_labels_all.mean():.4f}")
    print(f"Global Accuracy@0.5   : {global_acc_05:.4f}")
    print(f"Global mIoU@0.5       : {global_miou_05:.4f}")
    print(f"Global best mIoU      : {global_best_miou:.4f} @ best_miou_thr={global_best_miou_t:.3f}")
    print(f"Global F1@0.5         : {global_f1_05:.4f}")
    print(f"Global Recall@0.5     : {global_recall_05:.4f}")
    print(f"Global best F1        : {global_best_f1:.4f} @ best_f1_thr={global_best_f1_t:.3f}")
    print(f"Global Recall@best mIoU: {global_recall_best_miou:.4f} @ best_miou_thr={global_best_miou_t:.3f}")

    if infer_time_records:
        df_time = pd.DataFrame(infer_time_records)
        time_path = os.path.join(args.out_dir, "inference_time.csv")
        df_time.to_csv(time_path, index=False)
        total_model_infer_time = float(df_time["estimated_file_infer_time_sec"].sum())
        avg_file_infer_time = float(df_time["estimated_file_infer_time_sec"].mean())
        avg_point_infer_time_ms = float(df_time["infer_time_per_point_ms"].mean())
        throughput_points_per_sec = float(df_time["full_points"].sum() / max(total_model_infer_time, 1e-12))
        print("\n=== INFERENCE TIME SUMMARY ===")
        print(f"Files evaluated             : {len(df_time)}")
        print(f"Total wall eval time         : {total_eval_time:.6f} sec")
        print(f"Total model inference time   : {total_model_infer_time:.6f} sec")
        print(f"Avg model time per file      : {avg_file_infer_time:.6f} sec/file")
        print(f"Avg model time per point     : {avg_point_infer_time_ms:.6f} ms/point")
        print(f"Throughput                   : {throughput_points_per_sec:.2f} points/sec")
        print(f"Wrote inference timing CSV   : {time_path}")
    else:
        avg_file_infer_time = 0.0
        avg_point_infer_time_ms = 0.0
        total_model_infer_time = 0.0
        throughput_points_per_sec = 0.0

    eval_summary = {
        "num_files": len(results),
        "global_pos_ratio": float(full_labels_all.mean()),
        "global_acc_at_05": float(global_acc_05),
        "global_miou_at_05": float(global_miou_05),
        "global_best_miou": float(global_best_miou),
        "global_best_miou_thr": float(global_best_miou_t),
        "global_f1_at_05": float(global_f1_05),
        "global_recall_at_05": float(global_recall_05),
        "global_best_f1": float(global_best_f1),
        "global_best_f1_thr": float(global_best_f1_t),
        "global_recall_at_best_miou": float(global_recall_best_miou),
        "total_wall_eval_time_sec": float(total_eval_time),
        "total_model_infer_time_sec": float(total_model_infer_time),
        "avg_model_infer_time_sec_per_file": float(avg_file_infer_time),
        "avg_model_infer_time_ms_per_point": float(avg_point_infer_time_ms),
        "throughput_points_per_sec": float(throughput_points_per_sec),
    }
    summary_metrics_path = os.path.join(args.out_dir, "eval_summary_metrics.csv")
    pd.DataFrame([eval_summary]).to_csv(summary_metrics_path, index=False)
    print(f"Wrote eval summary metrics   : {summary_metrics_path}")

    df_sum = pd.DataFrame([r.__dict__ for r in results]).sort_values("best_miou", ascending=False)
    sum_path = os.path.join(args.out_dir, "summary_sampled.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"Wrote per-file summary: {sum_path}")

    report_path = os.path.join(args.out_dir, "experiment_report.md")
    write_experiment_report(args, eval_summary, report_path)
    print(f"Wrote experiment report: {report_path}")
    print(f"Wrote PLY point clouds: {pc_dir}")


def main():
    args = build_arg_parser().parse_args()
    run_full_eval(args)


if __name__ == "__main__":
    main()
