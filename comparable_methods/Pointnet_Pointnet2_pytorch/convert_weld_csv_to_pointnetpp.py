import os
import re
import glob
import json
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

# =========================
# 标签映射
# 如果原始 label 已经是 0/1，就不用改
# =========================
LABEL_MAP = {
    0: 0,
    1: 1,
    # 如果以后需要：
    # -1: 0,
    # 255: 1,
}

# =========================
# 文件名规则
# 示例:
# downsampled_labeled_1_aug_1_label_auto.csv
# 提取:
#   sample_id = 1
#   aug_id = 1
# =========================
FILE_ID_PATTERN = re.compile(
    r"downsampled_labeled_(\d+)_aug_(\d+)_label_auto\.csv$"
)

# 必需列
REQUIRED_COLS = ["x", "y", "z", "nx", "ny", "nz", "label"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_split_from_sample_id(sample_id: int):
    for split, ids in FOLD_1.items():
        if sample_id in ids:
            return split
    return None


def extract_ids(filename: str):
    """
    从文件名中提取 sample_id 和 aug_id
    例如:
    downsampled_labeled_1_aug_1_label_auto.csv
    -> (1, 1)
    """
    match = FILE_ID_PATTERN.search(filename)
    if not match:
        return None, None
    sample_id = int(match.group(1))
    aug_id = int(match.group(2))
    return sample_id, aug_id


def normalize_label_array(raw_label):
    """
    按 LABEL_MAP 映射标签；没有出现在映射表中的值保持原值
    """
    return np.vectorize(lambda x: LABEL_MAP.get(x, x))(raw_label).astype(np.int64)


def convert_one_csv(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")

    # 坐标
    coord = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # 法向
    normal = df[["nx", "ny", "nz"]].to_numpy(dtype=np.float32)

    # 颜色：你的数据没有 RGB，这里补零
    color = np.zeros((len(df), 3), dtype=np.float32)

    # 标签
    raw_label = df["label"].to_numpy()
    segment = normalize_label_array(raw_label)

    # 基本检查
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

    return {
        "num_points": int(n),
        "coord_shape": list(coord.shape),
        "normal_shape": list(normal.shape),
        "color_shape": list(color.shape),
        "segment_shape": list(segment.shape),
        "unique_labels": sorted(np.unique(segment).tolist()),
    }


def main():
    ensure_dir(OUT_ROOT)

    csv_files = sorted(glob.glob(os.path.join(CSV_ROOT, "*.csv")))
    print(f"Found {len(csv_files)} csv files.")

    split_counter = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    summary = {
        "csv_root": os.path.abspath(CSV_ROOT),
        "out_root": os.path.abspath(OUT_ROOT),
        "fold": "fold_1",
        "files": [],
    }

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        sample_id, aug_id = extract_ids(filename)

        if sample_id is None:
            print(f"[Skip] 无法从文件名提取 sample_id / aug_id: {filename}")
            split_counter["skipped"] += 1
            continue

        split = get_split_from_sample_id(sample_id)
        if split is None:
            print(f"[Skip] 样本编号 {sample_id} 不在 fold_1 划分里: {filename}")
            split_counter["skipped"] += 1
            continue

        # 输出目录名：保留原 stem，避免不同 aug 覆盖
        sample_name = Path(filename).stem
        out_dir = os.path.join(OUT_ROOT, split, sample_name)

        print(f"[{split}] sample_id={sample_id}, aug_id={aug_id} -> {sample_name}")

        try:
            stats = convert_one_csv(csv_path, out_dir)
            split_counter[split] += 1

            summary["files"].append({
                "filename": filename,
                "sample_name": sample_name,
                "sample_id": sample_id,
                "aug_id": aug_id,
                "split": split,
                "csv_path": os.path.abspath(csv_path),
                "out_dir": os.path.abspath(out_dir),
                **stats
            })

        except Exception as e:
            print(f"[Error] {filename}: {e}")
            split_counter["skipped"] += 1

    # 保存 summary
    meta_dir = os.path.join(OUT_ROOT, "meta")
    ensure_dir(meta_dir)
    with open(os.path.join(meta_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Summary:")
    for k, v in split_counter.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()