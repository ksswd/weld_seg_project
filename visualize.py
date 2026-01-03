import argparse
import os
from typing import Optional, TextIO

import numpy as np
import pandas as pd


def _normalize(v: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    hi = np.percentile(v, clip_percentile)
    lo = np.percentile(v, 100.0 - clip_percentile) if clip_percentile > 50 else np.min(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    v = np.clip(v, lo, hi)
    return (v - lo) / (hi - lo)


def colors_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (N,) 0/1
    returns colors: (N,3)
    """
    m = (mask.astype(np.int32) != 0)
    colors = np.zeros((mask.shape[0], 3), dtype=np.float32)
    colors[~m] = np.array([0.65, 0.65, 0.65], dtype=np.float32)  # light gray
    colors[m] = np.array([0.90, 0.15, 0.10], dtype=np.float32)   # red
    return colors


def colors_from_scalar(s: np.ndarray, cmap: str = "viridis", clip_percentile: float = 99.0) -> np.ndarray:
    """
    s: (N,) scalar values
    returns colors: (N,3)
    """
    s01 = _normalize(s, clip_percentile=clip_percentile)
    try:
        from matplotlib import colormaps
        rgba = colormaps.get_cmap(cmap)(s01)  # (N,4)
    except Exception:
        # Fallback for older matplotlib
        from matplotlib import cm
        rgba = cm.get_cmap(cmap)(s01)  # (N,4)
    return rgba[:, :3].astype(np.float32)


def colors_from_confusion(
    labels: np.ndarray,
    probs: np.ndarray,
    thr: float,
) -> np.ndarray:
    """
    Confusion-matrix coloring for binary classification.

    - TP (y=1, pred=1): green
    - FP (y=0, pred=1): red
    - FN (y=1, pred=0): blue
    - TN (y=0, pred=0): gray
    """
    y = (labels.astype(np.float32).reshape(-1) > 0.5)
    p = (probs.astype(np.float32).reshape(-1) > float(thr))
    tp = y & p
    fp = (~y) & p
    fn = y & (~p)
    tn = (~y) & (~p)
    colors = np.zeros((y.shape[0], 3), dtype=np.float32)
    colors[tn] = np.array([0.60, 0.60, 0.60], dtype=np.float32)  # gray
    colors[tp] = np.array([0.10, 0.85, 0.20], dtype=np.float32)  # green
    colors[fp] = np.array([0.90, 0.15, 0.10], dtype=np.float32)  # red
    colors[fn] = np.array([0.10, 0.35, 0.95], dtype=np.float32)  # blue
    return colors


def write_ply_xyz_rgb_ascii(
    path: str,
    xyz: np.ndarray,
    rgb01: np.ndarray,
    *,
    comment: Optional[str] = None,
) -> None:
    """
    Write a standard ASCII PLY file with per-vertex RGB (0-255).

    This avoids Open3D / OpenGL issues on remote servers.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb01.ndim != 2 or rgb01.shape[1] != 3:
        raise ValueError(f"rgb must be (N,3), got {rgb01.shape}")
    if xyz.shape[0] != rgb01.shape[0]:
        raise ValueError(f"xyz N ({xyz.shape[0]}) != rgb N ({rgb01.shape[0]})")

    rgb255 = np.clip(np.round(rgb01 * 255.0), 0, 255).astype(np.uint8)
    n = xyz.shape[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _w(f: TextIO, s: str) -> None:
        f.write(s + "\n")

    with open(path, "w", encoding="utf-8") as f:
        _w(f, "ply")
        _w(f, "format ascii 1.0")
        if comment:
            _w(f, f"comment {comment}")
        _w(f, f"element vertex {n}")
        _w(f, "property float x")
        _w(f, "property float y")
        _w(f, "property float z")
        _w(f, "property uchar red")
        _w(f, "property uchar green")
        _w(f, "property uchar blue")
        _w(f, "end_header")
        for p, c in zip(xyz, rgb255):
            _w(f, f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")


def build_xyz_and_colors(df: pd.DataFrame, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not {"x", "y", "z"}.issubset(df.columns):
        raise ValueError("CSV must contain x,y,z columns.")

    xyz = df[["x", "y", "z"]].values.astype(np.float32)
    if colors.shape[0] != xyz.shape[0]:
        raise ValueError(f"colors length {colors.shape[0]} != points length {xyz.shape[0]}")
    return xyz, colors.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Export CSV as a colored point cloud (.ply).")
    ap.add_argument("--csv", required=True, help="Path to CSV that contains x,y,z plus color columns.")
    ap.add_argument(
        "--color_by",
        default="mask",
        help="Column to color by (e.g. mask, abs_err_curvature_raw, pred_linearity). Use 'mask' for red/gray.",
    )
    ap.add_argument(
        "--confusion",
        action="store_true",
        help="Color by confusion matrix using --label_col and --prob_col (overrides --color_by).",
    )
    ap.add_argument("--label_col", default="label", help="Label column name for --confusion.")
    ap.add_argument("--prob_col", default="prob", help="Probability column name for --confusion.")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold for --confusion (prob > thr => positive).")
    ap.add_argument("--masked_only", action="store_true", help="Only keep points where mask==1 (if mask column exists).")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap for scalar coloring.")
    ap.add_argument("--clip_percentile", type=float, default=99.0, help="Percentile clipping for scalar color normalization.")
    ap.add_argument(
        "--save_ply",
        default="",
        help="Output .ply path (colored point cloud). If empty, write next to CSV with suffix.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.masked_only:
        if "mask" not in df.columns:
            raise ValueError("--masked_only requires 'mask' column in CSV")
        df = df[df["mask"].astype(int) == 1].reset_index(drop=True)

    if df.shape[0] == 0:
        raise ValueError("No points to visualize (empty dataframe after filtering).")

    if args.confusion:
        if args.label_col not in df.columns:
            raise ValueError(f"--label_col '{args.label_col}' not found in CSV.")
        if args.prob_col not in df.columns:
            raise ValueError(f"--prob_col '{args.prob_col}' not found in CSV.")
        colors = colors_from_confusion(df[args.label_col].values, df[args.prob_col].values, args.thr)
        color_tag = f"confusion_thr{args.thr:g}"
    elif args.color_by == "mask":
        if "mask" not in df.columns:
            raise ValueError("color_by=mask requires 'mask' column in CSV")
        colors = colors_from_mask(df["mask"].values)
        color_tag = "mask"
    else:
        if args.color_by not in df.columns:
            raise ValueError(f"Column '{args.color_by}' not found in CSV. Available: {list(df.columns)}")
        colors = colors_from_scalar(df[args.color_by].values, cmap=args.cmap, clip_percentile=args.clip_percentile)
        color_tag = args.color_by

    if args.save_ply:
        out_path = args.save_ply
    else:
        base = os.path.basename(args.csv).replace(".csv", "")
        suffix = f"{color_tag}"
        if args.masked_only:
            suffix += "_maskedOnly"
        out_path = os.path.join(os.path.dirname(args.csv), f"{base}_{suffix}.ply")

    xyz, rgb01 = build_xyz_and_colors(df, colors)
    write_ply_xyz_rgb_ascii(
        out_path,
        xyz,
        rgb01,
        comment=f"source={os.path.basename(args.csv)} color={color_tag}",
    )
    print(f"Saved colored point cloud to: {out_path}")


if __name__ == "__main__":
    main()

