import numpy as np
import os

label_dir = "data/processed/label"
files = [f for f in os.listdir(label_dir) if f.endswith('.npz')]

print("=" * 60)
print("Label Data Statistics")
print("=" * 60)

total_pos = 0
total_neg = 0

for fname in files:
    fpath = os.path.join(label_dir, fname)
    data = np.load(fpath)

    labels = data['labels'].reshape(-1)
    valid_mask = labels >= 0

    pos = int(((labels == 1) & valid_mask).sum())
    neg = int(((labels == 0) & valid_mask).sum())
    total_points = len(labels)
    valid_points = int(valid_mask.sum())

    total_pos += pos
    total_neg += neg

    print(f"\n{fname}:")
    print(f"  Total points: {total_points:,}")
    print(f"  Valid points: {valid_points:,}")
    print(f"  Positive (weld): {pos:,} ({100*pos/valid_points:.2f}%)")
    print(f"  Negative (non-weld): {neg:,} ({100*neg/valid_points:.2f}%)")
    print(f"  Ratio (neg/pos): {neg/pos:.2f}" if pos > 0 else "  Ratio: N/A")

print("\n" + "=" * 60)
print("Overall Statistics")
print("=" * 60)
print(f"Total positive: {total_pos:,}")
print(f"Total negative: {total_neg:,}")
print(f"Overall ratio (neg/pos): {total_neg/total_pos:.2f}" if total_pos > 0 else "N/A")
print(f"Positive percentage: {100*total_pos/(total_pos+total_neg):.2f}%")
