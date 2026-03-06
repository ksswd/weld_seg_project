#!/usr/bin/env python3
"""
Test script to compare self-supervised vs supervised pretraining modes.

Usage:
    python test_supervised_pretrain.py --mode self_supervised
    python test_supervised_pretrain.py --mode supervised
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from utils.config import Config


def create_test_config(mode):
    """Create a test config for the specified training mode."""
    # Create a copy of the base config
    test_config = Config()

    # Set training mode
    test_config.PRETRAIN_MODE = mode

    # Reduce epochs for testing
    test_config.NUM_EPOCHS = 10  # Just a few epochs for comparison

    # Use different weights save path
    test_config.WEIGHTS_SAVE_DIR = f"weights/test_{mode}"

    # Different log directory
    test_config.LOG_DIR = f"logs/test_{mode}"

    return test_config


def test_loss_modes():
    """Test both training modes with a simple batch."""
    print("Testing loss computation for both modes...")

    # Create test data
    B, N = 2, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock batch data
    batch = {
        'curvature': torch.randn(B, N, 1).abs() * 0.01 + 0.001,
        'coordinate': torch.randn(B, N, 3) * 10,
        'features': torch.randn(B, N, 8),
        'normals': torch.randn(B, N, 3),
        'principal_dir': torch.randn(B, N, 3),
        'local_density': torch.randn(B, N, 1).abs(),
        'linearity': torch.randn(B, N, 1).abs(),
        'mask': torch.ones(B, N, dtype=torch.bool)
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

    # Test both modes
    for mode in ['self_supervised', 'supervised']:
        print(f"\n--- Testing {mode} mode ---")

        # Temporarily set config
        Config.PRETRAIN_MODE = mode

        try:
            from train.pretrain import recon_criterion
            loss = recon_criterion(recon, batch, mask)
            print(f"✓ {mode} mode: loss = {loss.item():.6f}")
        except Exception as e:
            print(f"✗ {mode} mode failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Test supervised vs self-supervised pretraining')
    parser.add_argument('--mode', choices=['self_supervised', 'supervised'],
                       help='Training mode to test (optional - will test both if not specified)')
    parser.add_argument('--test_loss_only', action='store_true',
                       help='Only test loss computation, do not run full training')

    args = parser.parse_args()

    if args.test_loss_only:
        test_loss_modes()
        return

    if args.mode:
        print(f"Testing {args.mode} pretraining mode...")

        # Modify config temporarily
        original_mode = Config.PRETRAIN_MODE
        Config.PRETRAIN_MODE = args.mode

        try:
            # Import and run training
            from train.pretrain import run_pretrain
            run_pretrain(Config)
            print(f"\n✓ {args.mode} training completed!")
        except ImportError as e:
            print(f"Cannot run full training: {e}")
            print("Try --test_loss_only to test loss computation only")
        except Exception as e:
            print(f"\n✗ {args.mode} training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original config
            Config.PRETRAIN_MODE = original_mode
    else:
        print("No mode specified. Use --test_loss_only to test loss computation for both modes.")


if __name__ == "__main__":
    main()