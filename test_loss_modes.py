#!/usr/bin/env python3
"""
Test script to compare loss computation between self-supervised and supervised modes.
"""

import torch
import numpy as np
from utils.config import Config


def test_recon_criterion():
    """Test the recon_criterion function with both modes."""

    # Mock GlobalConfig
    class GlobalConfig:
        PRETRAIN_CURV_TARGET = "log"
        PRETRAIN_CURV_EPS = 1e-6
        PRETRAIN_RECON_WEIGHTS = [2.0, 2.0, 2.0, 2.0]
        PRETRAIN_USE_NORM_LOSS = True

    # Create test data
    B, N = 2, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock batch data
    batch = {
        'curvature': torch.randn(B, N, 1).abs() * 0.01 + 0.001,
        'coordinate': torch.randn(B, N, 3) * 10,
    }

    # Mock mask
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :70] = True  # 70% masked

    # Mock model output
    recon = torch.randn(B, N, 4)

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}
    mask = mask.to(device)
    recon = recon.to(device)

    def recon_criterion_test(recon, batch, mask, train_mode):
        """Test version of recon_criterion"""
        B, N, _ = recon.shape

        # Normalize curvature
        curv = batch['curvature']
        curv_mode = str(getattr(GlobalConfig, "PRETRAIN_CURV_TARGET", "log")).lower().strip()
        if curv_mode == "log":
            eps = float(getattr(GlobalConfig, "PRETRAIN_CURV_EPS", 1e-6))
            curv_t = torch.log(curv.clamp_min(0) + eps)
        else:
            curv_t = curv

        # Normalize coordinates (xyz)
        coord = batch['coordinate']  # (B, N, 3)
        coord_min = coord.min(dim=1, keepdim=True)[0].min(dim=0, keepdim=True)[0]  # (1, 1, 3)
        coord_max = coord.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]  # (1, 1, 3)
        coord_range = (coord_max - coord_min).clamp_min(1e-8)
        coord_t = (coord - coord_min) / coord_range  # (B, N, 3)

        # Concatenate targets: [curvature, x, y, z]
        gt = torch.cat([curv_t, coord_t], dim=-1)  # (B, N, 4)

        if train_mode == "supervised":
            # Supervised mode: Use ground truth directly as labels for masked points
            m = mask.unsqueeze(-1).float()  # (B, N, 1)
            num_masked = mask.sum().clamp(min=1).float()

            weights = torch.tensor(
                getattr(GlobalConfig, "PRETRAIN_RECON_WEIGHTS", [2.0, 2.0, 2.0, 2.0]),
                device=recon.device,
                dtype=recon.dtype
            ).reshape(1, 1, 4)

            diff = (recon - gt) * m  # Only penalize masked points
            per_channel_mse = (diff ** 2).sum(dim=(0, 1)) / num_masked  # (4,)

            loss = (per_channel_mse * weights.squeeze()).sum()

        elif train_mode == "self_supervised":
            # Self-supervised mode: Original reconstruction approach
            m = mask.unsqueeze(-1).float()  # (B, N, 1)
            num_masked = mask.sum().clamp(min=1).float()

            weights = torch.tensor(
                getattr(GlobalConfig, "PRETRAIN_RECON_WEIGHTS", [2.0, 2.0, 2.0, 2.0]),
                device=recon.device,
                dtype=recon.dtype
            ).reshape(1, 1, 4)

            diff = (recon - gt) * m
            use_norm_loss = getattr(GlobalConfig, "PRETRAIN_USE_NORM_LOSS", True)

            if use_norm_loss:
                masked_diff_sq = (diff ** 2).sum(dim=(0, 1))  # (4,)
                channel_rms = (masked_diff_sq / num_masked).sqrt().clamp_min(1e-8)  # (4,)
                rms_weights = 1.0 / channel_rms
                effective_weights = weights * rms_weights  # (4,)
                per_channel_mse = (diff ** 2).sum(dim=(0, 1)) / num_masked  # (4,)
            else:
                effective_weights = weights
                per_channel_mse = (diff ** 2).sum(dim=(0, 1)) / num_masked  # (4,)

            loss = (per_channel_mse * effective_weights.squeeze()).sum()

        return loss

    # Test both modes
    print("Testing loss computation for both modes...")
    print(f"Device: {device}")
    print(f"Batch size: {B}, Points per batch: {N}")
    print(f"Masked ratio: {mask.float().mean().item():.1%}")

    for mode in ['self_supervised', 'supervised']:
        print(f"\n--- Testing {mode} mode ---")

        try:
            loss = recon_criterion_test(recon, batch, mask, mode)
            print(f"✓ {mode} mode: loss = {loss.item():.6f}")

            # Test loss variation
            recon_test = recon + torch.randn_like(recon) * 0.1
            loss_test = recon_criterion_test(recon_test, batch, mask, mode)
            print(f"  Loss variation: {abs(loss.item() - loss_test.item()):.6f}")

        except Exception as e:
            print(f"✗ {mode} mode failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_recon_criterion()