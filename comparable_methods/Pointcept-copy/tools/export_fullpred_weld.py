#!/usr/bin/env python3
"""
Convert Pointcept weld predictions (*.npy) to *_fullpred.csv for visualization.

Example:
python comparable_methods/Pointcept/tools/export_fullpred_weld.py \
  --pred_dir comparable_methods/Pointcept/exp/weld/weld_100epoches/result \
  --data_root comparable_methods/Pointcept/data/weld \
  --split test \
  --out_dir comparable_methods/Pointcept/exp/weld/weld_100epoches/fullpred
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def find_sample_dir(data_root: str, split: str, sample_name: str) -> str:
    candidate = os.path.join(data_root, split, sample_name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"sample dir not found: {candidate}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Pointcept result dir containing *_pred.npy")
    ap.add_argument("--data_root", required=True, help="Pointcept data root, e.g. comparable_methods/Pointcept/data/weld")
    ap.add_argument("--split", default="test", help="split name under data_root")
    ap.add_argument("--out_dir", required=True, help="output dir for *_fullpred.csv")
    args = ap.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*_pred.npy")))
    if not pred_files:
        raise SystemExit(f"no *_pred.npy found in {args.pred_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    for pred_path in pred_files:
        base = os.path.basename(pred_path)
        sample_name = base.replace("_pred.npy", "")
        sample_dir = find_sample_dir(args.data_root, args.split, sample_name)

        coord = np.load(os.path.join(sample_dir, "coord.npy")).astype(np.float32)
        segment = np.load(os.path.join(sample_dir, "segment.npy")).astype(np.int32)
        pred = np.load(pred_path).astype(np.int32)

        if pred.shape[0] != coord.shape[0]:
            raise ValueError(f"size mismatch: {sample_name} pred={pred.shape} coord={coord.shape}")
        if segment.shape[0] != coord.shape[0]:
            raise ValueError(f"size mismatch: {sample_name} segment={segment.shape} coord={coord.shape}")

        df = pd.DataFrame(
            {
                "x": coord[:, 0],
                "y": coord[:, 1],
                "z": coord[:, 2],
                "label": segment,
                "prob": pred.astype(np.float32),
            }
        )
        out_csv = os.path.join(args.out_dir, f"{sample_name}_fullpred.csv")
        df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
