#!/usr/bin/env python3
"""Export PointNet++ weld predictions to *_fullpred.csv for unified evaluation."""

import argparse
import glob
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="pointnet2_sem_seg")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--npoint", type=int, default=4096)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--time_csv", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol-seed", type=int, default=20260823)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "models"))

    model_module = importlib.import_module(args.model)
    classifier = model_module.get_model(2)
    classifier.eval()

    ckpt_path = os.path.join(args.log_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    layout = ckpt.get("input_layout")
    if layout != "xyz_norm_normal_xyz_norm":
        raise ValueError(
            f"checkpoint has incompatible/unknown input_layout={layout!r}; "
            "retrain it with the fixed PointNet++ pipeline before unified export")
    classifier.load_state_dict(ckpt["model_state_dict"])

    split_dir = os.path.join(args.root, args.split)
    samples = sorted([d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
    if not samples:
        raise SystemExit(f"no samples under {split_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    timing_rows = []

    for sample_dir in samples:
        name = os.path.basename(sample_dir)
        coord = np.load(os.path.join(sample_dir, "coord.npy")).astype(np.float32)
        normal = np.load(os.path.join(sample_dir, "normal.npy")).astype(np.float32)
        seg = np.load(os.path.join(sample_dir, "segment.npy")).astype(np.int64)

        center = coord.mean(axis=0)
        xyz_centered = coord - center
        scale = np.max(np.linalg.norm(xyz_centered, axis=1))
        if scale < 1e-8:
            scale = 1.0
        xyz_norm = xyz_centered / scale
        # Must match WeldDataset: PointNet++ radius queries consume channels
        # 0:3, which therefore need to be normalized coordinates.
        points = np.concatenate([xyz_norm, normal, xyz_norm], axis=1).astype(np.float32)

        pred = np.zeros(len(points), dtype=np.int32)
        prob = np.zeros(len(points), dtype=np.float32)
        order = np.random.default_rng(args.protocol_seed).permutation(len(points))
        infer_time = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for start in range(0, len(points), args.npoint):
            end = min(start + args.npoint, len(points))
            valid_n = end - start
            valid_idx = order[start:end]
            chunk = points[valid_idx]
            if valid_n < args.npoint:
                pad_idx = np.arange(args.npoint - valid_n) % max(valid_n, 1)
                chunk = np.concatenate([chunk, chunk[pad_idx]], axis=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                pt = torch.from_numpy(chunk[None, ...]).to(device).transpose(2, 1)
                logits, _ = classifier(pt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time += time.perf_counter() - t0
            logits = logits.squeeze(0)[:valid_n]
            prob[valid_idx] = F.softmax(logits, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
            pred[valid_idx] = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        timing_rows.append({"sample": name, "num_points": int(coord.shape[0]), "npoint": int(args.npoint), "infer_time_sec": float(infer_time)})

        df = pd.DataFrame({
                "x": coord[:, 0],
                "y": coord[:, 1],
                "z": coord[:, 2],
                "label": seg.astype(np.int32),
            "pred": pred.astype(np.int32),
            "prob": prob.astype(np.float32),
        })
        out_csv = os.path.join(args.out_dir, f"{name}_fullpred.csv")
        df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")

    timing_df = pd.DataFrame(timing_rows)
    timing_csv = args.time_csv or os.path.join(args.out_dir, "inference_timing.csv")
    timing_df.to_csv(timing_csv, index=False)
    avg_time = float(timing_df["infer_time_sec"].mean()) if len(timing_df) else 0.0
    total_time = float(timing_df["infer_time_sec"].sum()) if len(timing_df) else 0.0
    print(f"[timing] saved: {timing_csv}")
    print(f"[timing] total_sec={total_time:.6f}, avg_sec={avg_time:.6f}, num_samples={len(timing_df)}")


if __name__ == "__main__":
    main()
