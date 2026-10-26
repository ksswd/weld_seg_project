#!/usr/bin/env python3
"""Count model parameters directly from weld benchmark checkpoints.

Run with an environment containing PyTorch, for example:
    conda run -n weld_seg python scripts/count_model_parameters.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


DEFAULT_CHECKPOINTS = {
    "Ours": "results/final/seed_points_sweep/ablation_geom_on_gate_on_seed444_pts4096/weights/best_finetune.pth",
    "DGCNN": "comparable_methods/dgcnn.pytorch/outputs/dgcnn_seed444_pts4096_train/models/best_model.pth",
    "PointNet++": "comparable_methods/Pointnet_Pointnet2_pytorch/log/seed_points_sweep/pointnet2_seed444_pts4096/checkpoints/best_model.pth",
    "PointMLP": "comparable_methods/pointMLP-pytorch/part_segmentation/checkpoints/pointMLP_pointmlp_seed444_pts4096/best_insiou_model.pth",
    "PointConv": "comparable_methods/pointconv_pytorch/log/seed_points_sweep/pointconv_seed444_pts4096/checkpoints/best_model.pth",
    "Point Transformer V3": "comparable_methods/Pointcept-copy/exp/weld_1_17/weld_newdata_200epochs/model/model_best.pth",
}

STATE_KEYS = ("model_state_dict", "state_dict", "model", "net", "network")
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dictionary, got {type(checkpoint).__name__}")
    for key in STATE_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value, key
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, "<root>"
    candidates = [k for k, v in checkpoint.items() if isinstance(v, dict)]
    raise ValueError(f"could not locate model state_dict; nested keys: {candidates}")


def count_checkpoint(name: str, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    # These are trusted checkpoints produced by this local project. Explicit
    # weights_only=False keeps compatibility with PyTorch 2.6+ and legacy
    # checkpoints that contain harmless training metadata objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state, state_key = extract_state_dict(checkpoint)
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    buffers = {k: v for k, v in tensors.items() if k.endswith(BUFFER_SUFFIXES)}
    parameters = {k: v for k, v in tensors.items() if k not in buffers}
    parameter_count = sum(v.numel() for v in parameters.values())
    buffer_count = sum(v.numel() for v in buffers.values())
    parameter_bytes = sum(v.numel() * v.element_size() for v in parameters.values())
    return {
        "method": name,
        "parameters": parameter_count,
        "parameters_m": parameter_count / 1_000_000,
        "buffer_elements": buffer_count,
        "parameter_size_mib": parameter_bytes / 1024**2,
        "checkpoint_size_mib": path.stat().st_size / 1024**2,
        "state_dict_key": state_key,
        "checkpoint": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, default=[])
    parser.add_argument("--output", type=Path, default=Path("results/model_parameters.csv"))
    args = parser.parse_args()
    models = dict(args.model) if args.model else {k: Path(v) for k, v in DEFAULT_CHECKPOINTS.items()}

    rows = [count_checkpoint(name, path) for name, path in models.items()]
    print(f"{'Method':<24} {'Parameters':>14} {'Params (M)':>12} {'FP size (MiB)':>15}")
    print("-" * 69)
    for row in rows:
        print(
            f"{row['method']:<24} {row['parameters']:>14,d} "
            f"{row['parameters_m']:>12.3f} {row['parameter_size_mib']:>15.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
