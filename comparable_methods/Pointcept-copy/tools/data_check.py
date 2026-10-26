import numpy as np

coord = np.load("data/weld/train/downsampled_labeled_3_aug_1_label_auto/coord.npy")
normal = np.load("data/weld/train/downsampled_labeled_3_aug_1_label_auto/normal.npy")
color = np.load("data/weld/train/downsampled_labeled_3_aug_1_label_auto/color.npy")
segment = np.load("data/weld/train/downsampled_labeled_3_aug_1_label_auto/segment.npy")

print(coord.shape, coord.dtype)
print(normal.shape, normal.dtype)
print(color.shape, color.dtype)
print(segment.shape, segment.dtype)
print(np.unique(segment))