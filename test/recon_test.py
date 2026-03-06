import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Ensure repo root is on sys.path so `model.*`, `train.*`, `utils.*` imports work
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.model import GeometryAwareTransformer
from train.mask_strategy import HighCurvatureMasker, RandomMasker
from train.block_segmenter import BlockSegmenter
from train.block_masker import BlockMasker
from utils.config import Config
from utils.io_utils import load_features_from_csv


def _subsample_np(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return arr[idx]


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def _stats(name: str, gt: np.ndarray, pred: np.ndarray) -> dict:
    err = (pred - gt).astype(np.float64)
    abs_err = np.abs(err)
    return {
        "name": name,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err * err).mean())),
        "p50_abs": float(np.percentile(abs_err, 50)),
        "p90_abs": float(np.percentile(abs_err, 90)),
        "p99_abs": float(np.percentile(abs_err, 99)),
        "corr": _pearson_corr(gt, pred),
    }


def compute_recon_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary metrics comparing gt vs pred for:
      - all points
      - masked points only (mask==1)
      - unmasked points only (mask==0)

    Supports both recon heads:
      - 1 channel: curvature only
      - 4 channels: [curvature, x, y, z]
    """
    if "mask" not in df.columns:
        raise ValueError("Missing required column 'mask' in recon_vis df")

    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "masked": (df["mask"].to_numpy(dtype=np.int32) == 1),
        "unmasked": (df["mask"].to_numpy(dtype=np.int32) == 0),
    }

    candidate_channels = [
        ("curvature_target", "gt_curvature_target", "pred_curvature_target"),
        ("x_target", "gt_x_target", "pred_x_target"),
        ("y_target", "gt_y_target", "pred_y_target"),
        ("z_target", "gt_z_target", "pred_z_target"),
        # backward compatibility with older exported files
        ("curvature_raw", "gt_curvature_raw", "pred_curvature_raw"),
        ("x", "gt_x", "pred_x"),
        ("y", "gt_y", "pred_y"),
        ("z", "gt_z", "pred_z"),
    ]
    channels = []
    for ch_name, gt_col, pred_col in candidate_channels:
        if gt_col in df.columns and pred_col in df.columns:
            if not (df[pred_col].isna().all() or df[gt_col].isna().all()):
                channels.append((ch_name, gt_col, pred_col))

    if not channels:
        raise ValueError("No valid gt/pred channel pairs found for metric computation")

    rows = []
    for subset_name, sel in subsets.items():
        if sel.sum() == 0:
            continue
        for ch_name, gt_col, pred_col in channels:
            rows.append({
                "subset": subset_name,
                "channel": ch_name,
                "n": int(sel.sum()),
                **_stats(
                    f"{subset_name}:{ch_name}",
                    df.loc[sel, gt_col].to_numpy(dtype=np.float32),
                    df.loc[sel, pred_col].to_numpy(dtype=np.float32),
                ),
            })
    return pd.DataFrame(rows)


def export_recon_csv(
    *,
    config: Config,
    csv_path: str,
    weights_path: str,
    out_dir: str,
    mask_ratio: float = 0.7,
    seed: Optional[int] = 0,
    max_points: Optional[int] = None,
    write_metrics: bool = True,
) -> str:
    """
    Run mask+recon pipeline close to train/pretrain.py, then export CSV for visualization.

    Supports recon head output channels:
      - 1 channel: [curvature_raw]
      - 4 channels: [curvature_raw, x, y, z]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    sample = load_features_from_csv(csv_path)

    n = sample["features"].shape[0]
    n_keep = int(max_points or getattr(config, "PRETRAIN_MAX_POINTS", 2048) or n)
    n_keep = min(n_keep, n)
    if n_keep < n:
        idx = np.random.choice(n, size=n_keep, replace=False)
        for k, v in list(sample.items()):
            if v is None:
                sample[k] = None
            else:
                sample[k] = _subsample_np(v, idx)

    # add batch dim: (1, N, C)
    features = torch.from_numpy(sample["features"][None, ...]).to(device)
    coordinate = torch.from_numpy(sample["coordinate"][None, ...]).to(device)
    normals = torch.from_numpy(sample["normals"][None, ...]).to(device)
    principal_dir = torch.from_numpy(sample["principal_dir"][None, ...]).to(device)
    curvature = torch.from_numpy(sample["curvature"][None, ...]).to(device)
    local_density = torch.from_numpy(sample["local_density"][None, ...]).to(device)
    linearity = torch.from_numpy(sample["linearity"][None, ...]).to(device)

    valid_mask = torch.ones(features.shape[:2], device=device, dtype=torch.bool)  # (1, N)

    # Use the same mask family as pretraining (prefer block-wise when enabled)
    use_block_mask = bool(getattr(config, "USE_BLOCK_MASK", False))
    if use_block_mask:
        segmenter = BlockSegmenter(
            target_points_per_block=getattr(config, "TARGET_POINTS_PER_BLOCK", 1000),
            high_curv_threshold=getattr(config, "HIGH_CURV_THRESHOLD", 0.01),
            min_high_curv_points=getattr(config, "MIN_HIGH_CURV_POINTS", 5),
            align_grid=True,
            grid_align_base=getattr(config, "GRID_ALIGN_BASE", 0.001),
        )
        block_masker = BlockMasker(
            mask_ratio=mask_ratio,
            strategy=getattr(config, "BLOCK_MASK_STRATEGY", "mixed"),
            weld_mask_ratio=getattr(config, "WELD_MASK_RATIO", None),
            bg_mask_ratio=getattr(config, "BG_MASK_RATIO", None),
            seed=seed,
        )

        valid_indices = torch.nonzero(valid_mask[0], as_tuple=False).squeeze(-1).cpu().numpy()
        points = coordinate[0, valid_indices].detach().cpu().numpy()
        curv_np = curvature[0, valid_indices].detach().cpu().numpy()
        blocks, block_labels = segmenter.segment(points, curv_np)
        blocks_original = [[int(valid_indices[idx]) for idx in block] for block in blocks]

        mask_1d = block_masker.generate_mask(blocks_original, block_labels, epoch=0).to(device)
        mask = mask_1d.unsqueeze(0).bool() & valid_mask  # (1, N)
    else:
        mask_type = str(getattr(config, "MASK_TYPE", "random")).lower()
        if mask_type == "random":
        masker = RandomMasker(mask_ratio=mask_ratio, seed=seed)
        elif mask_type == "curvature":
    masker = HighCurvatureMasker(mask_ratio=mask_ratio)
    else:
            masker = RandomMasker(mask_ratio=mask_ratio, seed=seed)
    mask = masker.generate_mask(curvature, valid_mask=valid_mask).squeeze(-1).bool()  # (1, N)
    mask = mask & valid_mask

    # Mirror train/pretrain.py masking
    masked_feat = features.clone()
    masked_feat[mask] = 0.0
    masked_curv = curvature.clone()
    masked_dens = local_density.clone()
    masked_lin = linearity.clone()
    masked_normals = normals.clone()
    masked_principal = principal_dir.clone()
    masked_curv[mask] = 0.0
    masked_dens[mask] = 0.0
    masked_lin[mask] = 0.0
    masked_normals[mask] = 0.0
    masked_principal[mask] = 0.0

    model = GeometryAwareTransformer(config).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    use_amp = bool(getattr(config, "USE_AMP", False)) and device.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                recon = model(
                    masked_feat,
                    coordinate,
                    masked_principal,
                    masked_curv,
                    masked_dens,
                    masked_normals,
                    masked_lin,
                    task="recon",
                )
        else:
            recon = model(
                masked_feat,
                coordinate,
                masked_principal,
                masked_curv,
                masked_dens,
                masked_normals,
                masked_lin,
                task="recon",
            )

    # Build export dataframe on CPU
    xyz = coordinate.squeeze(0).detach().cpu().numpy()
    m = mask.squeeze(0).detach().cpu().numpy().astype(np.int32)
    
    gt_curv_raw = curvature.squeeze(0).detach().cpu().numpy()[:, 0]
    pred = recon.squeeze(0).detach().cpu().numpy()
    out_dim = int(pred.shape[-1])

    # Build targets in the SAME space as train/pretrain.py::recon_criterion
    curv_mode = str(getattr(config, "PRETRAIN_CURV_TARGET", "log")).lower().strip()
    eps = float(getattr(config, "PRETRAIN_CURV_EPS", 1e-6))
    if curv_mode == "log":
        gt_curv_target = np.log(np.clip(gt_curv_raw, 0.0, None) + eps).astype(np.float32)
    else:
        gt_curv_target = gt_curv_raw.astype(np.float32)

    pred_curv_target = pred[:, 0].astype(np.float32)

    data = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "mask": m,
        # target-space columns (used for metrics)
        "gt_curvature_target": gt_curv_target,
        "pred_curvature_target": pred_curv_target,
        # raw-space columns (used for visualization)
        "gt_curvature_raw": gt_curv_raw,
    }

    # Curvature raw-space prediction (for visualization only)
    if curv_mode == "log":
        pred_curv_raw = np.exp(pred[:, 0].astype(np.float64)) - eps
        pred_curv_raw = np.clip(pred_curv_raw, 0.0, None).astype(np.float32)
    else:
        pred_curv_raw = pred[:, 0].astype(np.float32)
    data["pred_curvature_raw"] = pred_curv_raw
    
    # Optional xyz channels when recon head outputs 4 dims
    if out_dim >= 4:
    coord_min = xyz.min(axis=0, keepdims=True).astype(np.float64)
    coord_max = xyz.max(axis=0, keepdims=True).astype(np.float64)
    coord_range = np.maximum(coord_max - coord_min, 1e-8)
    
        gt_xyz_target = ((xyz.astype(np.float64) - coord_min) / coord_range).astype(np.float32)
        pred_xyz_target = pred[:, 1:4].astype(np.float32)

        pred_xyz_raw = (pred[:, 1:4].astype(np.float64) * coord_range + coord_min).astype(np.float32)

        data.update(
            {
                # target-space columns (used for metrics)
                "gt_x_target": gt_xyz_target[:, 0],
                "gt_y_target": gt_xyz_target[:, 1],
                "gt_z_target": gt_xyz_target[:, 2],
                "pred_x_target": pred_xyz_target[:, 0],
                "pred_y_target": pred_xyz_target[:, 1],
                "pred_z_target": pred_xyz_target[:, 2],
                # raw-space columns (used for visualization)
                "gt_x": xyz[:, 0],
                "gt_y": xyz[:, 1],
                "gt_z": xyz[:, 2],
            "pred_x": pred_xyz_raw[:, 0],
            "pred_y": pred_xyz_raw[:, 1],
            "pred_z": pred_xyz_raw[:, 2],
        }
    )

    df = pd.DataFrame(data)
    # target-space errors (for reliable quantitative analysis)
    df["abs_err_curvature_target"] = np.abs(df["pred_curvature_target"] - df["gt_curvature_target"])
    if out_dim >= 4:
        df["abs_err_x_target"] = np.abs(df["pred_x_target"] - df["gt_x_target"])
        df["abs_err_y_target"] = np.abs(df["pred_y_target"] - df["gt_y_target"])
        df["abs_err_z_target"] = np.abs(df["pred_z_target"] - df["gt_z_target"])

    # raw-space errors (for visual inspection)
    df["abs_err_curvature_raw"] = np.abs(df["pred_curvature_raw"] - df["gt_curvature_raw"])
    if out_dim >= 4:
    df["abs_err_x"] = np.abs(df["pred_x"] - df["gt_x"])
    df["abs_err_y"] = np.abs(df["pred_y"] - df["gt_y"])
    df["abs_err_z"] = np.abs(df["pred_z"] - df["gt_z"])

    base = os.path.basename(csv_path).replace(".csv", "")
    out_path = os.path.join(out_dir, f"{base}_recon_vis.csv")
    df.to_csv(out_path, index=False)

    if write_metrics:
        mdf = compute_recon_metrics(df)
        metrics_path = os.path.join(out_dir, f"{base}_recon_metrics.csv")
        mdf.to_csv(metrics_path, index=False)
        masked = mdf[mdf["subset"] == "masked"]
        if len(masked):
            print(f"[metrics] masked-only summary for {base}:")
            for _, r in masked.iterrows():
                print(
                    f"  - {r['channel']}: mae={r['mae']:.6g} rmse={r['rmse']:.6g} "
                    f"p90={r['p90_abs']:.6g} corr={r['corr']:.3f} (n={int(r['n'])})"
                )
        print(f"[metrics] wrote: {metrics_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, default=None, help="A processed_csv file to visualize.")
    ap.add_argument("--weights", required=False, default="weights/best_pretrain.pth")
    ap.add_argument("--out_dir", required=False, default="data/predictions/recon_vis")
    ap.add_argument("--mask_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_points", type=int, default=0, help="0 means use Config.PRETRAIN_MAX_POINTS")
    ap.add_argument("--no_metrics", action="store_true", help="Do not write *_recon_metrics.csv")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        root = Config.PROCESSED_DATA_DIR
        candidates = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".csv") and "_pred" not in f
        ]
        if not candidates:
            raise FileNotFoundError(f"No csv files found under {root}")
        csv_path = sorted(candidates)[0]

    out_path = export_recon_csv(
        config=Config,
        csv_path=csv_path,
        weights_path=args.weights,
        out_dir=args.out_dir,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        max_points=(args.max_points if args.max_points and args.max_points > 0 else None),
        write_metrics=(not args.no_metrics),
    )
    print(f"Saved recon visualization CSV: {out_path}")


if __name__ == "__main__":
    main()
