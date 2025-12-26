import numpy as np
import os

print("=" * 80)
print("Checking Training vs Test Data Consistency")
print("=" * 80)

# Check training data
train_dir = "data/processed/label"
test_dir = "data/test/new"

print("\n1. Training data files:")
train_files = [f for f in os.listdir(train_dir) if f.endswith('.npz') and '_pred' not in f]
for f in train_files:
    print(f"   {f}")

print("\n2. Test data files:")
if os.path.exists(test_dir):
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
    for f in test_files:
        print(f"   {f}")
else:
    print(f"   Test directory does not exist: {test_dir}")
    test_files = []

# Load a training sample and check feature statistics
if train_files:
    print("\n3. Training data feature statistics:")
    train_path = os.path.join(train_dir, train_files[0])
    train_data = np.load(train_path)
    train_feat = train_data['features']
    print(f"   File: {train_files[0]}")
    print(f"   Shape: {train_feat.shape}")
    print(f"   Mean: {train_feat.mean(axis=0)}")
    print(f"   Std:  {train_feat.std(axis=0)}")

    if 'labels' in train_data:
        labels = train_data['labels']
        print(f"   Labels shape: {labels.shape}")
        print(f"   Positive points: {(labels == 1).sum()} ({100*(labels == 1).sum()/len(labels):.2f}%)")

# Load a test sample and check feature statistics
if test_files:
    print("\n4. Test data feature statistics:")
    test_path = os.path.join(test_dir, test_files[0])
    test_data = np.load(test_path)
    test_feat = test_data['features']
    print(f"   File: {test_files[0]}")
    print(f"   Shape: {test_feat.shape}")
    print(f"   Mean: {test_feat.mean(axis=0)}")
    print(f"   Std:  {test_feat.std(axis=0)}")

# Check if normalization is consistent
if train_files and test_files:
    print("\n5. Normalization consistency check:")
    train_mean = train_feat.mean(axis=0)
    test_mean = test_feat.mean(axis=0)
    diff = np.abs(train_mean - test_mean)
    print(f"   Mean difference: {diff}")
    if np.any(diff > 0.5):
        print("   ⚠️  WARNING: Large difference detected! Features may not be normalized consistently!")
    else:
        print("   ✓ Features appear to be normalized consistently")

print("\n" + "=" * 80)
