import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure repo root is on sys.path so `model.*`, `train.*`, `utils.*` imports work
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.model import GeometryAwareTransformer
from utils.config import Config
from utils.io_utils import load_features_from_csv
from utils.fold_data_split import collect_files_by_fold


@dataclass
class FileEvalResult:
    file: str
    n_points: int
    pos_ratio: float
    covered_ratio: float
    f1_at_05: float
    best_f1: float
    best_thr: float


def grouped_split(files: List[str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Match finetune.py logic: split by group key = basename with `_augXX.csv` stripped.
    We take the last val_ratio of sorted group keys as validation.
    """
    groups: Dict[str, List[str]] = {}
    for p in sorted(files):
        base = os.path.basename(p)
        key = re.sub(r"_aug\\d+\\.csv$", ".csv", base)
        groups.setdefault(key, []).append(p)
    keys = sorted(groups.keys())
    n_val_groups = max(1, int(val_ratio * len(keys)))
    val_keys = set(keys[-n_val_groups:])
    train_files = [p for k in keys if k not in val_keys for p in groups[k]]
    val_files = [p for k in keys if k in val_keys for p in groups[k]]
    return train_files, val_files


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
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def miou_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, fn, tn = _confusion(y_true, y_pred)
    iou_pos = tp / max(tp + fp + fn, 1)
    iou_neg = tn / max(tn + fp + fn, 1)
    return float((iou_pos + iou_neg) / 2.0)


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


def collate_fn_fps(batch):
    max_points = int(getattr(Config, "MAX_POINTS", 1024))
    sampled = []
    file_paths = []
    orig_sizes = []
    orig_indices = []
    for item in batch:
        if item.get("labels", None) is None:
            raise KeyError("This CSV has no 'label' column. Full-eval expects *_label_*.csv files.")
        n = item["coordinate"].shape[0]
        k = min(max_points, n)
        from utils.downsampling import fps_with_cache as fps
        idx = fps(item["coordinate"], k)
        labels_soft_src = item.get("labels_soft", None)
        if labels_soft_src is None:
            labels_soft_src = item["labels"]
        sampled.append({
            "features": item["features"][idx],
            "coordinate": item["coordinate"][idx],
            "normals": item["normals"][idx],
            "principal_dir": item["principal_dir"][idx],
            "curvature": item["curvature"][idx],
            "local_density": item["local_density"][idx],
            "linearity": item["linearity"][idx],
            "labels": item["labels"][idx],
            "labels_soft": labels_soft_src[idx],
        })
        file_paths.append(item["file_path"])
        orig_sizes.append(int(n))
        orig_indices.append(idx)

    b = len(sampled)
    max_n = max(s["features"].shape[0] for s in sampled)
    max_n = min(max_n, max_points)

    def pad(arr_list, shape):
        out = np.full(shape, 0.0, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            n = min(arr.shape[0], shape[1])
            out[i, :n] = arr[:n]
        return out

    feats = pad([s["features"] for s in sampled], (b, max_n, sampled[0]["features"].shape[1]))
    coordinate = pad([s["coordinate"] for s in sampled], (b, max_n, 3))
    normals = pad([s["normals"] for s in sampled], (b, max_n, 3))
    principal = pad([s["principal_dir"] for s in sampled], (b, max_n, 3))
    curvature = pad([s["curvature"] for s in sampled], (b, max_n, 1))
    density = pad([s["local_density"] for s in sampled], (b, max_n, 1))
    linearity = pad([s["linearity"] for s in sampled], (b, max_n, 1))
    labels = pad([s["labels"] for s in sampled], (b, max_n, 1))
    labels_soft = pad([s["labels_soft"] for s in sampled], (b, max_n, 1))

    mask = torch.zeros(b, max_n, dtype=torch.bool)
    for i, s in enumerate(sampled):
        mask[i, : min(s["features"].shape[0], max_n)] = True

    return {
        "features": torch.from_numpy(feats),
        "coordinate": torch.from_numpy(coordinate),
        "normals": torch.from_numpy(normals),
        "principal_dir": torch.from_numpy(principal),
        "curvature": torch.from_numpy(curvature),
        "local_density": torch.from_numpy(density),
        "linearity": torch.from_numpy(linearity),
        "labels": torch.from_numpy(labels),
        "labels_soft": torch.from_numpy(labels_soft),
        "mask": mask,
        "file_paths": file_paths,
        "orig_sizes": orig_sizes,
        "orig_indices": orig_indices,
    }


@torch.no_grad()
def infer_probs_snapshot(
    model: GeometryAwareTransformer,
    device: torch.device,
    batch: dict,
    use_amp: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    batch = {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    logits = model(
        batch["features"],
        batch["coordinate"],
        batch["principal_dir"],
        batch["curvature"],
        batch["local_density"],
        batch["normals"],
        batch["linearity"],
        task="class",
    )
    probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
    labels = batch["labels"].squeeze(-1).detach().cpu().numpy()
    mask = batch["mask"].detach().cpu().numpy()

    probs_list, labels_list = [], []
    for i in range(probs.shape[0]):
        vm = mask[i].astype(bool)
        probs_list.append(probs[i][vm].astype(np.float32))
        labels_list.append(labels[i][vm].astype(np.int32))
    return probs_list, labels_list


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Full point-cloud evaluation via chunked inference coverage.")
    ap.add_argument("--weights", default="weights/best_finetune.pth", help="Path to finetuned checkpoint.")
    ap.add_argument("--data_dir", default=Config.PROCESSED_DATA_DIR, help="Directory containing *_label_*.csv")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Group-level validation ratio.")
    ap.add_argument("--max_points", type=int, default=1024, help="Points per inference chunk.")
    ap.add_argument("--passes", type=int, default=1, help="Number of random cover passes to average (>=1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/predictions/full_eval", help="Where to write per-file prediction CSVs.")
    ap.add_argument("--export_simple_pred", action="store_true", help="Also export x,y,z,prediction only CSV for each file.")
    ap.add_argument("--simple_thr", type=float, default=0.5, help="Threshold used for simple prediction export.")
    ap.add_argument("--use_fold_split", action="store_true", help="Use group-aware fold split instead of grouped val_ratio split.")
    ap.add_argument("--fold_id", type=str, default="fold_1", choices=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"], help="Fold id when use_fold_split=True")
    return ap


def run_full_eval(args):
    if args.use_fold_split:
        eval_files = collect_files_by_fold(
            mode='test',
            fold_id=args.fold_id,
            data_dir=args.data_dir,
            include_labeled_aug=True,
            strict=False,
            print_stats=True,
        )
        print(f"Eval files from fold split: {len(eval_files)} (fold={args.fold_id}, mode=test)")
    else:
        labeled_files = sorted(
            [
                os.path.join(args.data_dir, f)
                for f in os.listdir(args.data_dir)
                if f.endswith(".csv") and "_label_" in f
            ]
        )
        if not labeled_files:
            raise FileNotFoundError(f"No *_label_*.csv found under {args.data_dir}")

        _, eval_files = grouped_split(labeled_files, val_ratio=args.val_ratio)
        print(f"Eval files: {len(eval_files)} (grouped split, val_ratio={args.val_ratio})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryAwareTransformer(Config).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    use_amp = bool(getattr(Config, "USE_AMP", False)) and device.type == "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    dataset = WeldDataset(eval_files)
    loader = DataLoader(dataset, batch_size=int(getattr(Config, "BATCH_SIZE", 1)), shuffle=False, collate_fn=collate_fn_fps, pin_memory=True)

    results: List[FileEvalResult] = []

    sampled_probs_all = []
    sampled_labels_all = []

    for batch in loader:
        file_paths = batch.pop("file_paths")
        orig_sizes = batch.pop("orig_sizes")
        orig_indices = batch.pop("orig_indices")
        sampled_xyz_batch = batch["coordinate"].detach().cpu().numpy()
        probs_list, labels_list = infer_probs_snapshot(model, device, batch, use_amp)

        for file_path, probs, labels, n_orig, idx, sampled_xyz in zip(file_paths, probs_list, labels_list, orig_sizes, orig_indices, sampled_xyz_batch):
            covered = float(len(idx) / max(int(n_orig), 1))
            f1_05 = f1_score_numpy(labels, (probs > 0.5).astype(np.int32))
            miou_05 = miou_score_numpy(labels, (probs > 0.5).astype(np.int32))
            best_miou, best_t = best_threshold_miou(probs, labels)
            best_f1, _ = best_threshold_f1(probs, labels)

            results.append(
                FileEvalResult(
                    file=os.path.basename(file_path),
                    n_points=int(labels.shape[0]),
                    pos_ratio=float(labels.mean()),
                    covered_ratio=float(covered),
                    f1_at_05=float(f1_05),
                    best_f1=float(best_f1),
                    best_thr=float(best_t),
                )
            )

            df_sampled = pd.DataFrame({
                "sampled_index": np.asarray(idx, dtype=np.int32),
                "prob": probs,
                "label": labels,
            })
            out_sampled = os.path.join(args.out_dir, os.path.basename(file_path).replace(".csv", "_sampled_fullpred.csv"))
            df_sampled.to_csv(out_sampled, index=False)

            if args.export_simple_pred:
                simple_pred = (probs > float(args.simple_thr)).astype(np.int32)
                df_simple = pd.DataFrame({
                    "sampled_index": np.asarray(idx, dtype=np.int32),
                    "prediction": simple_pred,
                })
                out_simple = os.path.join(args.out_dir, os.path.basename(file_path).replace(".csv", "_sampled_pred_simple.csv"))
                df_simple.to_csv(out_simple, index=False)

                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(sampled_xyz[:, 0], sampled_xyz[:, 1], sampled_xyz[:, 2], c=probs, s=1, cmap="viridis")
                fig.colorbar(sc, ax=ax, shrink=0.6, label="prob")
                ax.set_title(os.path.basename(file_path))
                plt.tight_layout()
                vis_path = os.path.join(vis_dir, os.path.basename(file_path).replace(".csv", "_prob.png"))
                plt.savefig(vis_path, dpi=200)
                plt.close(fig)

            # full-size回填到原始长度（未采样点记为-1）
            full_df = pd.read_csv(file_path)
            full_pred = np.full((len(full_df),), -1, dtype=np.int32)
            full_pred[np.asarray(idx, dtype=np.int32)] = (probs > float(args.simple_thr)).astype(np.int32)
            full_df["pred"] = full_pred
            out_full = os.path.join(args.out_dir, os.path.basename(file_path).replace(".csv", "_fullpred.csv"))
            full_df.to_csv(out_full, index=False)

            print(
                f"[{os.path.basename(file_path)}] sampledN={labels.size} pos={labels.mean():.4f} "
                f"samplemIoU@0.5={miou_05:.4f} sampleBestmIoU={best_miou:.4f} best_miou_thr={best_t:.3f} | "
                f"sampleF1@0.5={f1_05:.4f} sampleBestF1={best_f1:.4f}"
            )

            sampled_probs_all.append(probs)
            sampled_labels_all.append(labels)

    sampled_probs_all = np.concatenate(sampled_probs_all)
    sampled_labels_all = np.concatenate(sampled_labels_all)
    global_sample_f1_05 = f1_score_numpy(sampled_labels_all, (sampled_probs_all > 0.5).astype(np.int32))
    global_sample_best_f1, global_sample_best_t = best_threshold_f1(sampled_probs_all, sampled_labels_all)

    global_sample_miou_05 = miou_score_numpy(sampled_labels_all, (sampled_probs_all > 0.5).astype(np.int32))
    global_sample_best_miou, global_sample_best_miou_t = best_threshold_miou(sampled_probs_all, sampled_labels_all)

    print("\n=== SUMMARY (sampled mIoU-first) ===")
    print(f"Global pos ratio      : {sampled_labels_all.mean():.4f}")
    print(f"Global mIoU@0.5      : {global_sample_miou_05:.4f}")
    print(f"Global best mIoU     : {global_sample_best_miou:.4f} @ best_miou_thr={global_sample_best_miou_t:.3f}")
    print(f"Global F1@0.5        : {global_sample_f1_05:.4f}")
    print(f"Global best F1       : {global_sample_best_f1:.4f} @ best_f1_thr={global_sample_best_t:.3f}")

    df_sum = pd.DataFrame([r.__dict__ for r in results]).sort_values("best_f1", ascending=False)
    sum_path = os.path.join(args.out_dir, "summary_sampled.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"Wrote per-file summary: {sum_path}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_full_eval(args)


if __name__ == "__main__":
    main()




