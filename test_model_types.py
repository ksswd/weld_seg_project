#!/usr/bin/env python3
"""
Test script to compare Transformer vs MLP model performance.
"""

import torch
import numpy as np
from utils.config import Config
from model.model import GeometryAwareTransformer


def test_model_types():
    """Test both model types with the same input."""

    # Test configurations
    configs = {
        'transformer': Config(),
        'mlp': Config()
    }
    configs['transformer'].MODEL_TYPE = 'transformer'
    configs['mlp'].MODEL_TYPE = 'mlp'

    # Create test input
    B, N = 2, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = torch.randn(B, N, 8).to(device)
    coordinate = torch.randn(B, N, 3).to(device)
    principal_dir = torch.randn(B, N, 3).to(device)
    curvature = torch.randn(B, N, 1).to(device)
    density = torch.randn(B, N, 1).to(device)
    normals = torch.randn(B, N, 3).to(device)
    linearity = torch.randn(B, N, 1).to(device)

    print(f"Testing on device: {device}")
    print(f"Batch size: {B}, Points: {N}")
    print(f"Input feature dims: {features.shape}")

    for model_type, config in configs.items():
        print(f"\n--- Testing {model_type.upper()} model ---")

        try:
            # Create model
            model = GeometryAwareTransformer(config).to(device)
            model.eval()

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")

            # Test forward pass
            with torch.no_grad():
                # Test classification
                class_output = model(features, coordinate, principal_dir, curvature,
                                   density, normals, linearity, task='class')
                print(f"Classification output shape: {class_output.shape}")

                # Test reconstruction
                recon_output = model(features, coordinate, principal_dir, curvature,
                                   density, normals, linearity, task='recon')
                print(f"Reconstruction output shape: {recon_output.shape}")

            print(f"✓ {model_type.upper()} model works correctly!")

        except Exception as e:
            print(f"✗ {model_type.upper()} model failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_model_types()