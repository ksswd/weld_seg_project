import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _extract_aug_index(path: Path) -> int:
    match = re.search(r"_aug_(\d+)", path.stem)
    if not match:
        raise ValueError(f"无法从文件名解析增强编号: {path.name}")
    return int(match.group(1))


def split_and_copy(input_dir: Path, output_dir: Path, train_n: int, val_n: int, test_n: int, exts):
    files_by_base = defaultdict(list)

    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        name = p.stem
        if "_aug_" not in name:
            continue
        base = name.split("_aug_")[0]
        files_by_base[base].append(p)

    total = 0
    for base, files in sorted(files_by_base.items()):
        files = sorted(files, key=_extract_aug_index)
        expected = train_n + val_n + test_n
        if len(files) < expected:
            raise ValueError(f"{base} 的预处理文件数量不足：{len(files)} < {expected}")

        plan = (
            ("train", files[:train_n]),
            ("validation", files[train_n:train_n + val_n]),
            ("test", files[train_n + val_n:train_n + val_n + test_n]),
        )

        for split_name, split_files in plan:
            split_dir = output_dir / split_name
            ensure_dir(split_dir)
            for src in split_files:
                dst = split_dir / src.name
                shutil.copy2(src, dst)
                total += 1

        print(f"{base}: train={train_n}, validation={val_n}, test={test_n}")

    print(f"完成，共复制 {total} 个文件。")


def parse_args():
    parser = argparse.ArgumentParser(description="将预处理后的点云按固定比例复制到 train/validation/test")
    parser.add_argument("--input_dir", default="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/new/processed_data", help="预处理数据目录")
    parser.add_argument("--output_dir", default="/lab/shiyh_lab/12432730/code/weld_seg_project/weld_seg_project/data/new/dataset_processed", help="目标数据集目录")
    parser.add_argument("--train_n", type=int, default=14, help="每个原始点云分到训练集的文件数")
    parser.add_argument("--val_n", type=int, default=3, help="每个原始点云分到验证集的文件数")
    parser.add_argument("--test_n", type=int, default=3, help="每个原始点云分到测试集的文件数")
    parser.add_argument("--exts", nargs="*", default=[".csv"], help="要复制的文件后缀，如 .ply .csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_and_copy(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
        exts={e.lower() for e in args.exts},
    )
