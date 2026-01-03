import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure repo root is on sys.path so `model.*`, `train.*`, `utils.*` imports work
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


def f1_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


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


@torch.no_grad()
def infer_probs_full(
    model: GeometryAwareTransformer,
    device: torch.device,
    sample: dict,
    max_points: int,
    passes: int,
    seed: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce per-point probabilities for all points by covering the full set with chunked inference.
    Each pass uses a new random permutation; we aggregate logits per point and average.

    Returns:
      probs: (N,) float32
      counts: (N,) int32 number of times each point was predicted
    """
    features = sample["features"]
    coordinate = sample["coordinate"]
    normals = sample["normals"]
    principal_dir = sample["principal_dir"]
    curvature = sample["curvature"]
    local_density = sample["local_density"]
    linearity = sample["linearity"]

    n = features.shape[0]
    sum_logits = np.zeros((n,), dtype=np.float64)
    counts = np.zeros((n,), dtype=np.int32)

    rng = np.random.default_rng(seed)

    for p in range(int(passes)):
        perm = rng.permutation(n)
        for start in range(0, n, max_points):
            idx = perm[start:start + max_points]
            # build tensors (1, K, C)
            f = torch.from_numpy(features[idx][None, ...]).to(device)
            coor = torch.from_numpy(coordinate[idx][None, ...]).to(device)
            nor = torch.from_numpy(normals[idx][None, ...]).to(device)
            prin = torch.from_numpy(principal_dir[idx][None, ...]).to(device)
            curv = torch.from_numpy(curvature[idx][None, ...]).to(device)
            dens = torch.from_numpy(local_density[idx][None, ...]).to(device)
            lin = torch.from_numpy(linearity[idx][None, ...]).to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(f, coor, prin, curv, dens, nor, lin, task="class").squeeze(0).squeeze(-1)
            else:
                logits = model(f, coor, prin, curv, dens, nor, lin, task="class").squeeze(0).squeeze(-1)

            logits_np = logits.detach().cpu().numpy().astype(np.float32)
            sum_logits[idx] += logits_np
            counts[idx] += 1

    # avoid div-by-zero
    avg_logits = (sum_logits / np.maximum(counts, 1)).astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    return probs.astype(np.float32), counts


def main():
    ap = argparse.ArgumentParser(description="Full point-cloud evaluation via chunked inference coverage.")
    ap.add_argument("--weights", default="weights/best_finetune.pth", help="Path to finetuned checkpoint.")
    ap.add_argument("--data_dir", default=Config.PROCESSED_DATA_DIR, help="Directory containing *_label_*.csv")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Group-level validation ratio.")
    ap.add_argument("--max_points", type=int, default=1024, help="Points per inference chunk.")
    ap.add_argument("--passes", type=int, default=1, help="Number of random cover passes to average (>=1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/predictions/full_eval", help="Where to write per-file prediction CSVs.")
    args = ap.parse_args()

    labeled_files = sorted(
        [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith(".csv") and "_label_" in f
        ]
    )
    if not labeled_files:
        raise FileNotFoundError(f"No *_label_*.csv found under {args.data_dir}")

    _, val_files = grouped_split(labeled_files, val_ratio=args.val_ratio)
    print(f"Val files: {len(val_files)} (grouped split, val_ratio={args.val_ratio})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryAwareTransformer(Config).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    use_amp = bool(getattr(Config, "USE_AMP", False)) and device.type == "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    all_probs = []
    all_labels = []
    results: List[FileEvalResult] = []

    for file_path in val_files:
        # load model inputs
        sample = load_features_from_csv(file_path)
        if sample.get("labels", None) is None:
            raise ValueError(f"CSV has no label column: {file_path}")
        labels = sample["labels"].reshape(-1).astype(np.int32)

        probs, counts = infer_probs_full(
            model=model,
            device=device,
            sample=sample,
            max_points=int(args.max_points),
            passes=int(args.passes),
            seed=int(args.seed),
            use_amp=use_amp,
        )

        covered = (counts > 0).mean().item()
        f1_05 = f1_score_numpy(labels, (probs > 0.5).astype(np.int32))
        best_f1, best_t = best_threshold_f1(probs, labels)

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

        # export per-file prediction CSV for visualization/post-processing
        df_full = pd.read_csv(file_path, usecols=["x", "y", "z", "label"])
        df_full["prob"] = probs
        df_full["count"] = counts
        out_csv = os.path.join(args.out_dir, os.path.basename(file_path).replace(".csv", "_fullpred.csv"))
        df_full.to_csv(out_csv, index=False)
        print(
            f"[{os.path.basename(file_path)}] N={labels.size} pos={labels.mean():.4f} "
            f"covered={covered:.3f} F1@0.5={f1_05:.4f} bestF1={best_f1:.4f} thr={best_t:.3f} -> {out_csv}"
        )

        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    global_f1_05 = f1_score_numpy(all_labels, (all_probs > 0.5).astype(np.int32))
    global_best_f1, global_best_t = best_threshold_f1(all_probs, all_labels)

    print("\n=== SUMMARY (val set aggregated) ===")
    print(f"Global pos ratio: {all_labels.mean():.4f}")
    print(f"Global F1@0.5   : {global_f1_05:.4f}")
    print(f"Global best F1  : {global_best_f1:.4f} @ thr={global_best_t:.3f}")

    # write summary table
    df_sum = pd.DataFrame([r.__dict__ for r in results]).sort_values("best_f1", ascending=False)
    sum_path = os.path.join(args.out_dir, "summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"Wrote per-file summary: {sum_path}")


if __name__ == "__main__":
    main()




