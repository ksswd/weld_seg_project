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
from train.mask_strategy import HighCurvatureMasker
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
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
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
    """
    required = [
        "mask",
        "gt_curvature_raw", "gt_local_density_raw", "gt_linearity",
        "pred_curvature_raw", "pred_local_density_raw", "pred_linearity",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in recon_vis df: {missing}")

    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "masked": (df["mask"].to_numpy(dtype=np.int32) == 1),
        "unmasked": (df["mask"].to_numpy(dtype=np.int32) == 0),
    }

    channels = [
        ("curvature_raw", "gt_curvature_raw", "pred_curvature_raw"),
        ("local_density_raw", "gt_local_density_raw", "pred_local_density_raw"),
        ("linearity", "gt_linearity", "pred_linearity"),
    ]

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
    Run the same mask+recon pipeline as pretraining, then export a CSV for visualization.

    Recon head outputs 3 channels: [curvature_raw, local_density_raw, linearity].
    The exported CSV contains:
      - xyz
      - mask (0/1)
      - gt_* and pred_* for the 3 targets
      - abs_err_* for quick sanity checking
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
    masker = HighCurvatureMasker(mask_ratio=mask_ratio)
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
    model.load_state_dict(state)
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
    gt = np.concatenate(
        [
            curvature.squeeze(0).detach().cpu().numpy(),
            local_density.squeeze(0).detach().cpu().numpy(),
            linearity.squeeze(0).detach().cpu().numpy(),
        ],
        axis=-1,
    )
    pred = recon.squeeze(0).detach().cpu().numpy()

    # Convert predictions from normalized space back to raw space for visualization/metrics
    # Curvature: log-space -> raw
    curv_mode = str(getattr(config, "PRETRAIN_CURV_TARGET", "raw")).lower().strip()
    if curv_mode == "log":
        eps = float(getattr(config, "PRETRAIN_CURV_EPS", 1e-6))
        pred_curv_raw = np.exp(pred[:, 0].astype(np.float64)) - eps
        pred_curv_raw = np.clip(pred_curv_raw, 0.0, None).astype(np.float32)
    else:
        pred_curv_raw = pred[:, 0].astype(np.float32)
    
    # Density: norm-space -> raw (using current sample's min/max for denormalization)
    dens_mode = str(getattr(config, "PRETRAIN_DENSITY_TARGET", "norm")).lower().strip()
    gt_dens = gt[:, 1].astype(np.float64)
    if dens_mode == "norm":
        dens_min = float(np.min(gt_dens))
        dens_max = float(np.max(gt_dens))
        dens_range = max(dens_max - dens_min, 1e-8)
        pred_dens_raw = pred[:, 1].astype(np.float64) * dens_range + dens_min
        pred_dens_raw = np.clip(pred_dens_raw, dens_min, dens_max).astype(np.float32)
    elif dens_mode == "log":
        pred_dens_raw = np.exp(pred[:, 1].astype(np.float64)) - 1.0
        pred_dens_raw = np.clip(pred_dens_raw, 0.0, None).astype(np.float32)
    else:
        pred_dens_raw = pred[:, 1].astype(np.float32)
    
    # Linearity: norm-space -> raw (using current sample's min/max for denormalization)
    lin_mode = str(getattr(config, "PRETRAIN_LINEARITY_TARGET", "norm")).lower().strip()
    gt_lin = gt[:, 2].astype(np.float64)
    if lin_mode == "norm":
        lin_min = float(np.min(gt_lin))
        lin_max = float(np.max(gt_lin))
        lin_range = max(lin_max - lin_min, 1e-8)
        pred_lin_raw = pred[:, 2].astype(np.float64) * lin_range + lin_min
        pred_lin_raw = np.clip(pred_lin_raw, lin_min, lin_max).astype(np.float32)
    else:
        pred_lin_raw = pred[:, 2].astype(np.float32)

    df = pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "mask": m,
            "gt_curvature_raw": gt[:, 0],
            "gt_local_density_raw": gt[:, 1],
            "gt_linearity": gt[:, 2],
            "pred_curvature_raw": pred_curv_raw,
            "pred_local_density_raw": pred_dens_raw,
            "pred_linearity": pred_lin_raw,
        }
    )
    df["abs_err_curvature_raw"] = np.abs(df["pred_curvature_raw"] - df["gt_curvature_raw"])
    df["abs_err_local_density_raw"] = np.abs(df["pred_local_density_raw"] - df["gt_local_density_raw"])
    df["abs_err_linearity"] = np.abs(df["pred_linearity"] - df["gt_linearity"])

    base = os.path.basename(csv_path).replace(".csv", "")
    out_path = os.path.join(out_dir, f"{base}_recon_vis.csv")
    df.to_csv(out_path, index=False)

    if write_metrics:
        mdf = compute_recon_metrics(df)
        metrics_path = os.path.join(out_dir, f"{base}_recon_metrics.csv")
        mdf.to_csv(metrics_path, index=False)
        # Print a compact masked-only summary to console
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

    # Default to any one processed csv if not specified
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