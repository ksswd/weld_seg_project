import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# 路径配置
# =========================
CSV_ROOT = "data/raw"
OUT_ROOT = "data/weld"

# =========================
# fold_1 划分
# =========================
FOLD_1 = {
    "test": [1, 5, 9, 14],
    "val": [2, 6, 10, 17],
    "train": [3, 4, 7, 8, 11, 12, 13, 15, 16, 18, 19],
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
# 示例: downsampled_labeled_1_aug_1_label_auto.csv -> 1
FILE_ID_PATTERN = re.compile(r"downsampled_labeled_(\d+)_aug_\d+_label_auto\.csv$")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_split_from_sample_id(sample_id: int):
    for split, ids in FOLD_1.items():
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

    required_cols = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "curvature",
        "density",
        "linearity",
        "principal_dir_x",
        "principal_dir_y",
        "principal_dir_z",
        "label",
        "label_soft",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    # 坐标
    coord = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # 法向
    normal = df[["nx", "ny", "nz"]].to_numpy(dtype=np.float32)

    # 几何特征
    curvature = df[["curvature"]].to_numpy(dtype=np.float32)
    density = df[["density"]].to_numpy(dtype=np.float32)
    linearity = df[["linearity"]].to_numpy(dtype=np.float32)
    principal_dir = df[["principal_dir_x", "principal_dir_y", "principal_dir_z"]].to_numpy(dtype=np.float32)

    # 颜色：你没有 rgb，这里补零
    color = np.zeros((len(df), 3), dtype=np.float32)

    # 标签
    raw_label = df["label"].to_numpy()
    segment = np.vectorize(lambda x: LABEL_MAP.get(x, x))(raw_label).astype(np.int64)
    label_soft = df["label_soft"].to_numpy(dtype=np.float32)

    # 检查
    n = len(df)
    assert coord.shape == (n, 3), f"coord shape error: {coord.shape}"
    assert normal.shape == (n, 3), f"normal shape error: {normal.shape}"
    assert color.shape == (n, 3), f"color shape error: {color.shape}"
    assert curvature.shape == (n, 1), f"curvature shape error: {curvature.shape}"
    assert density.shape == (n, 1), f"density shape error: {density.shape}"
    assert linearity.shape == (n, 1), f"linearity shape error: {linearity.shape}"
    assert principal_dir.shape == (n, 3), f"principal_dir shape error: {principal_dir.shape}"
    assert segment.shape == (n,), f"segment shape error: {segment.shape}"
    assert label_soft.shape == (n,), f"label_soft shape error: {label_soft.shape}"

    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "coord.npy"), coord)
    np.save(os.path.join(out_dir, "normal.npy"), normal)
    np.save(os.path.join(out_dir, "color.npy"), color)
    np.save(os.path.join(out_dir, "curvature.npy"), curvature)
    np.save(os.path.join(out_dir, "density.npy"), density)
    np.save(os.path.join(out_dir, "linearity.npy"), linearity)
    np.save(os.path.join(out_dir, "principal_dir.npy"), principal_dir)
    np.save(os.path.join(out_dir, "segment.npy"), segment)
    np.save(os.path.join(out_dir, "label_soft.npy"), label_soft)


def main():
    ensure_dir(OUT_ROOT)

    csv_files = sorted(glob.glob(os.path.join(CSV_ROOT, "*.csv")))
    print(f"Found {len(csv_files)} csv files.")

    split_counter = {"train": 0, "val": 0, "test": 0, "skipped": 0}

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        sample_id = extract_sample_id(filename)

        if sample_id is None:
            print(f"[Skip] 无法从文件名提取样本编号: {filename}")
            split_counter["skipped"] += 1
            continue

        split = get_split_from_sample_id(sample_id)
        if split is None:
            print(f"[Skip] 样本编号 {sample_id} 不在 fold_1 划分里: {filename}")
            split_counter["skipped"] += 1
            continue

        sample_name = Path(filename).stem
        out_dir = os.path.join(OUT_ROOT, split, sample_name)

        print(f"[{split}] sample_id={sample_id} -> {sample_name}")
        convert_one_csv(csv_path, out_dir)
        split_counter[split] += 1

    print("\nDone.")
    print("Summary:")
    for k, v in split_counter.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()