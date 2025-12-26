#!/usr/bin/env python
"""Quick test to verify Focal Loss implementation"""
import torch
import sys
sys.path.insert(0, '.')
from utils.focal_loss import FocalLoss

print("=" * 60)
print("Testing Focal Loss Implementation")
print("=" * 60)

# Test case: extreme imbalance (24:1)
alpha = 0.04  # pos_ratio = 4%
gamma = 2.0

focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

# Scenario 1: Perfect prediction
print("\n1. Perfect prediction (all correct):")
logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])  # high confidence
targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
loss = focal(logits, targets)
print(f"   Loss: {loss.item():.6f} (should be ~0)")

# Scenario 2: Wrong prediction
print("\n2. Wrong prediction (all incorrect):")
logits = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])  # high confidence wrong
targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
loss = focal(logits, targets)
print(f"   Loss: {loss.item():.6f} (should be high)")

# Scenario 3: Uncertain prediction
print("\n3. Uncertain prediction (logits near 0):")
logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
loss = focal(logits, targets)
print(f"   Loss: {loss.item():.6f} (should be moderate)")

# Scenario 4: Realistic batch (imbalanced)
print("\n4. Realistic batch (24 negative, 1 positive):")
logits = torch.randn(1, 25) * 0.5  # random predictions
targets = torch.zeros(1, 25)
targets[0, 0] = 1.0  # only first point is positive
loss = focal(logits, targets)
print(f"   Loss: {loss.item():.6f} (should be 0.2-1.0)")

# Compare with BCE
import torch.nn as nn
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([24.0]), reduction='mean')
bce_loss = bce(logits, targets)
print(f"   BCE Loss (pos_weight=24): {bce_loss.item():.6f}")
print(f"   Focal Loss (alpha=0.04): {loss.item():.6f}")
print(f"   Ratio: {bce_loss.item() / (loss.item() + 1e-8):.2f}x")

print("\n" + "=" * 60)
print("Focal Loss test completed!")
print("If BCE loss >> Focal loss, Focal is working correctly.")
print("=" * 60)
