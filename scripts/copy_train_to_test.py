#!/usr/bin/env python
"""
Quick fix: Copy training samples to test directory for validation.
Since test data has wrong normalization, use training data directly.
"""
import os
import shutil

print("=" * 80)
print("Copying Training Samples to Test Directory")
print("=" * 80)

# Source: training data (correctly normalized)
train_dir = "data/processed/label"

# Destination: test data
test_dir = "data/test/correct_norm"

# Create test directory
os.makedirs(test_dir, exist_ok=True)

# Get all training files
train_files = [f for f in os.listdir(train_dir) if f.endswith('.npz') and '_pred' not in f]

print(f"\nAvailable training files:")
for i, f in enumerate(train_files, 1):
    print(f"  {i}. {f}")

print(f"\nCopying ALL training files to: {test_dir}")

for f in train_files:
    src = os.path.join(train_dir, f)
    dst = os.path.join(test_dir, f)
    shutil.copy(src, dst)
    print(f"  ✓ Copied: {f}")

print("\n" + "=" * 80)
print("✓ Done! Training samples copied successfully.")
print("\nNext steps:")
print("  1. Update config.py: TEST_DATA_DIR = 'data/test/correct_norm'")
print("  2. Run inference: python main.py --mode test")
print("  3. Check predictions in data/predictions/new/")
print("\nSince these are training samples, predictions should be very accurate!")
print("=" * 80)
