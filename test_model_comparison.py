#!/usr/bin/env python3
"""
Compare Transformer vs MLP model performance on pretraining.
"""

import os
import sys
import torch
from utils.config import Config


def create_test_config(model_type):
    """Create test config for model comparison."""
    config = Config()

    # Set model type
    config.MODEL_TYPE = model_type

    # Reduce epochs for testing
    config.NUM_EPOCHS = 50  # Just a few epochs for comparison

    # Use different directories
    config.WEIGHTS_SAVE_DIR = f"weights/test_{model_type}"
    config.LOG_DIR = f"logs/test_{model_type}"

    # Ensure directories exist
    os.makedirs(config.WEIGHTS_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    return config


def run_comparison_test():
    """Run training comparison between Transformer and MLP."""

    print("=== Model Comparison Test ===")
    print("Testing Transformer vs MLP on pretraining task")
    print("Each model will train for 5 epochs")
    print()

    results = {}

    for model_type in ['transformer', 'mlp']:
        print(f"--- Training {model_type.upper()} model ---")

        # Create config
        config = create_test_config(model_type)

        # Count model parameters
        try:
            from model.model import GeometryAwareTransformer
            model = GeometryAwareTransformer(config)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
        except Exception as e:
            print(f"Failed to create model: {e}")
            continue

        # Run training
        try:
            # Import training function
            from train.pretrain import run_pretrain
            print(f"Starting training for {config.NUM_EPOCHS} epochs...")

            # Run training
            run_pretrain(config)

            print(f"✓ {model_type.upper()} training completed!")
            results[model_type] = "success"

        except ImportError as e:
            print(f"Cannot run full training: {e}")
            print("Try running individual tests instead")
            results[model_type] = "import_error"
        except Exception as e:
            print(f"✗ {model_type.upper()} training failed: {e}")
            results[model_type] = "failed"

        print()

    # Summary
    print("=== Results Summary ===")
    for model_type, status in results.items():
        status_icon = "✓" if status == "success" else "✗"
        print(f"{status_icon} {model_type.upper()}: {status}")

    if all(status == "success" for status in results.values()):
        print("\nBoth models completed training successfully!")
        print("Compare the logs and saved weights to see performance differences.")
        print("\nTo test reconstruction performance:")
        print("1. Run: python test/recon_test.py --csv your_data.csv")
        print("2. Change MODEL_TYPE in config.py and re-run")
        print("3. Compare the reconstruction errors")
    else:
        print("\nSome models failed. Check the error messages above.")


if __name__ == "__main__":
    run_comparison_test()