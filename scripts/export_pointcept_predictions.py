#!/usr/bin/env python3
"""Convert Pointcept result *_pred.npy files to unified full-prediction CSVs."""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    split_dir = os.path.join(args.data_root, args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    exported = 0
    for name in sorted(os.listdir(split_dir)):
        sample_dir = os.path.join(split_dir, name)
        if not os.path.isdir(sample_dir):
            continue
        pred_path = os.path.join(args.result_dir, f"{name}_pred.npy")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"missing Pointcept prediction: {pred_path}")
        coord = np.load(os.path.join(sample_dir, "coord.npy"))
        label = np.load(os.path.join(sample_dir, "segment.npy")).reshape(-1)
        pred = np.load(pred_path).reshape(-1)
        if not (len(coord) == len(label) == len(pred)):
            raise ValueError(
                f"length mismatch for {name}: coord={len(coord)}, label={len(label)}, pred={len(pred)}"
            )
        pd.DataFrame(
            {
                "x": coord[:, 0].astype(np.float32),
                "y": coord[:, 1].astype(np.float32),
                "z": coord[:, 2].astype(np.float32),
                "label": label.astype(np.int8),
                "pred": pred.astype(np.int8),
            }
        ).to_csv(os.path.join(args.out_dir, f"{name}_fullpred.csv"), index=False)
        exported += 1
    if exported == 0:
        raise RuntimeError(f"no samples found under {split_dir}")
    print(f"Exported {exported} Pointcept full-prediction CSV files to {args.out_dir}")


if __name__ == "__main__":
    main()
