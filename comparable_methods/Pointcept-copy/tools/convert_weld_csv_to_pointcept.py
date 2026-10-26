import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# 路径配置
# =========================
CSV_ROOT = "/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/new/processed_csv_5"
OUT_ROOT = "data/weld_flod_1"

# =========================
# fold_1 划分
# =========================
# FOLD_1 = {
#     "test": [1, 5, 9, 14],
#     "val": [2, 6, 10, 17],
#     "train": [3, 4, 7, 8, 11, 12, 13, 15, 16, 18, 19],
# }
FOLD_1 = {
        "test": [1, 5, 9, 14],
        "val": [2, 6, 10, 17],
        "train": [3, 4, 7, 8, 11, 12, 13, 15, 16, 18, 19],
    }
FOLD_2 = {
        "test": [2, 6, 10, 17],
        "val": [3, 7, 11, 18],
        "train": [1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 19],
    }
FOLD_3 = {
        "test": [3, 7, 11, 18],
        "val": [4, 8, 12, 19],
        "train": [1, 2, 5, 6, 9, 10, 13, 14, 15, 16, 17],
    }
FOLD_4 = {
        "test": [4, 8, 12, 19],
        "val": [13, 15, 16],
        "train": [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 17, 18],
    }
FOLD_5 = {
        "test": [13, 15, 16],
        "val": [1, 5, 9, 14],
        "train": [2, 3, 4, 6, 7, 8, 10, 11, 12, 17, 18, 19],
    }
FOLD_6 = {
        "test": [3, 11],
        "val": [1, 2, 9, 10],
        "train": [1, 2, 9, 10],
    }

FOLD_MAP = {
    1: FOLD_1,
    2: FOLD_2,
    3: FOLD_3,
    4: FOLD_4,
    5: FOLD_5,
    6: FOLD_6,
}

# 如果你的 label 不是 0/1，可以在这里改
LABEL_MAP = {
    0: 0,
    1: 1,
    # 例如:
    # -1: 0,
    # 255: 1,
}

# 用于从文件名里提取样本编号
# 兼容两种命名：
# 1) downsampled_labeled_1_aug_1_label_auto.csv -> 1
# 2) downsampled_labeled_1_2_aug_1_label_auto.csv -> 1（取第一段作为样本编号）
FILE_ID_PATTERN = re.compile(r"downsampled_labeled_(\d+)(?:_\d+)?_aug_\d+_label_auto\.csv$")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_split_from_sample_id(sample_id: int, fold_split: dict):
    for split, ids in fold_split.items():
        if sample_id in ids:
            return split
    return None


def extract_sample_id(filename: str):
    match = FILE_ID_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1))


def convert_one_csv(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    required_cols = ["x", "y", "z", "nx", "ny", "nz", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    # 坐标
    coord = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # 法向
    normal = df[["nx", "ny", "nz"]].to_numpy(dtype=np.float32)

    # 颜色：你没有 rgb，这里补零
    color = np.zeros((len(df), 3), dtype=np.float32)

    # 标签
    raw_label = df["label"].to_numpy()
    segment = np.vectorize(lambda x: LABEL_MAP.get(x, x))(raw_label).astype(np.int64)

    # 检查
    n = len(df)
    assert coord.shape == (n, 3), f"coord shape error: {coord.shape}"
    assert normal.shape == (n, 3), f"normal shape error: {normal.shape}"
    assert color.shape == (n, 3), f"color shape error: {color.shape}"
    assert segment.shape == (n,), f"segment shape error: {segment.shape}"

    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "coord.npy"), coord)
    np.save(os.path.join(out_dir, "normal.npy"), normal)
    np.save(os.path.join(out_dir, "color.npy"), color)
    np.save(os.path.join(out_dir, "segment.npy"), segment)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert weld CSV to Pointcept npy format")
    parser.add_argument("--fold", type=int, default=1, choices=sorted(FOLD_MAP.keys()), help="选择使用哪个 fold 划分")
    parser.add_argument("--csv-root", type=str, default=CSV_ROOT, help="输入 CSV 根目录")
    parser.add_argument("--out-root", type=str, default=OUT_ROOT, help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    fold_idx = args.fold
    fold_split = FOLD_MAP[fold_idx]

    ensure_dir(args.out_root)

    csv_files = sorted(glob.glob(os.path.join(args.csv_root, "*.csv")))
    print(f"Using fold_{fold_idx}")
    print(f"Found {len(csv_files)} csv files.")

    split_counter = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        sample_id = extract_sample_id(filename)

        if sample_id is None:
            print(f"[Skip] 无法从文件名提取样本编号: {filename}")
            split_counter["skipped"] += 1
            continue

        split = get_split_from_sample_id(sample_id, fold_split)
        if split is None:
            print(f"[Skip] 样本编号 {sample_id} 不在 fold_{fold_idx} 划分里: {filename}")
            split_counter["skipped"] += 1
            continue

        sample_name = Path(filename).stem
        out_dir = os.path.join(args.out_root, split, sample_name)

        print(f"[{split}] sample_id={sample_id} -> {sample_name}")
        convert_one_csv(csv_path, out_dir)
        split_counter[split] += 1

    print("\nDone.")
    print("Summary:")
    for k, v in split_counter.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()