import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


FOLD_SPLITS: Dict[str, Dict[str, List[int]]] = {
    "fold_1": {
        "test": [1, 5, 9, 14],
        "val": [2, 6, 10, 17],
        "train": [3, 4, 7, 8, 11, 12, 13, 15, 16, 18, 19],
    },
    "fold_2": {
        "test": [2, 6, 10, 17],
        "val": [3, 7, 11, 18],
        "train": [1, 4, 5, 8, 9, 12, 13, 14, 15, 16, 19],
    },
    "fold_3": {
        "test": [3, 7, 11, 18],
        "val": [4, 8, 12, 19],
        "train": [1, 2, 5, 6, 9, 10, 13, 14, 15, 16, 17],
    },
    "fold_4": {
        "test": [4, 8, 12, 19],
        "val": [13, 15, 16],
        "train": [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 17, 18],
    },
    "fold_5": {
        "test": [13, 15, 16],
        "val": [1, 5, 9, 14],
        "train": [2, 3, 4, 6, 7, 8, 10, 11, 12, 17, 18, 19],
    },
     "fold_6": {
        "test": [3, 11],
        "val": [1, 2, 9, 10],
        "train": [1, 2, 9, 10],
    },
}


@dataclass
class IndexedFile:
    path: str
    filename: str
    group_id: int
    sample_type: str  # unlabeled_aug | labeled_original | labeled_aug | unknown


def get_fold_groups(fold_id: str) -> Dict[str, List[int]]:
    if fold_id not in FOLD_SPLITS:
        raise ValueError(f"Unknown fold_id: {fold_id}. Available: {list(FOLD_SPLITS.keys())}")
    split = FOLD_SPLITS[fold_id]
    _check_group_disjoint(split)
    return split


def parse_group_id(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    patterns = [
        r"^downsampled_weld_(\d+)_aug_\d+$",
        r"^downsampled_labeled_(\d+)_label_.*$",
        r"^downsampled_labeled_(\d+)_aug_\d+_label_.*$",
        r"^downsampled_labeled_(\d+)_\d+_label_.*$",
        r"^downsampled_labeled_(\d+)_\d+_aug_\d+_label_.*$",
        r"^(\d+)\s*-\s*(\d+).*$",
        r"^(\d+)-\d+$",
        r"^(\d+)-labeled$",
        r"^(\d+)-\d+-labeled$",
        r"^(\d+)_.*$",
        r"^(\d+)$",
    ]
    for p in patterns:
        m = re.match(p, stem)
        if m:
            return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse group_id from filename: {filename}")


def detect_sample_type(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if re.match(r"^downsampled_weld_\d+_aug_\d+$", stem):
        return "unlabeled_aug"
    # 兼容新的命名："1 - 1Cloud_aug_2_label_auto" / "10 - Cloud_aug_1_label_auto"
    if re.match(r"^\d+\s*-\s*.*label_.*$", stem):
        return "labeled_original"
    if re.match(r"^\d+-.*label_.*$", stem):
        return "labeled_original"
    # 标注原始样本（历史命名兼容）
    if re.match(r"^downsampled_labeled_\d+_label_.*$", stem):
        return "labeled_original"
    if re.match(r"^downsampled_labeled_\d+_\d+_label_.*$", stem):
        return "labeled_original"
    # 标注增强样本（历史命名兼容）
    if re.match(r"^downsampled_labeled_\d+_aug_\d+_label_.*$", stem):
        return "labeled_aug"
    if re.match(r"^downsampled_labeled_\d+_\d+_aug_\d+_label_.*$", stem):
        return "labeled_aug"
    if re.match(r"^\d+-\d+$", stem):
        return "unlabeled_aug"
    if re.match(r"^\d+-labeled$", stem):
        return "labeled_original"
    if re.match(r"^\d+-\d+-labeled$", stem):
        return "labeled_aug"
    return "unknown"


def scan_and_index_files(data_dir: str, strict: bool = True) -> List[IndexedFile]:
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and "_pred" not in f
    ]
    files.sort()
    indexed: List[IndexedFile] = []
    unknown: List[str] = []
    for p in sorted(files):
        fn = os.path.basename(p)
        try:
            gid = parse_group_id(fn)
            st = detect_sample_type(fn)
            if st == "unknown":
                unknown.append(fn)
            indexed.append(IndexedFile(path=p, filename=fn, group_id=gid, sample_type=st))
        except Exception as e:
            if strict:
                raise ValueError(f"Failed to index file '{fn}': {e}") from e
    if unknown and strict:
        raise ValueError(
            "Unknown sample type files detected (adjust naming rules or set strict=False):\n"
            + "\n".join(unknown[:20])
        )
    return indexed


def collect_files_by_fold(
    mode: str,
    fold_id: str,
    data_dir: str,
    include_labeled_aug: bool = False,
    include_labeled_in_pretrain: bool = False,
    strict: bool = True,
    print_stats: bool = True,
) -> List[str]:
    split = get_fold_groups(fold_id)
    indexed = scan_and_index_files(data_dir, strict=strict)

    mode_to_groups = {
        "pretrain": set(split["train"]),
        "finetune_train": set(split["train"]),
        "finetune_val": set(split["val"]),
        "test": set(split["test"]),
    }
    if mode not in mode_to_groups:
        raise ValueError(f"Unknown mode: {mode}")
    target_groups = mode_to_groups[mode]

    def allowed_type(st: str) -> bool:
        if mode == "pretrain":
            if include_labeled_in_pretrain:
                return st in ("unlabeled_aug", "labeled_original", "labeled_aug")
            return st == "unlabeled_aug"
        if mode == "finetune_val":
            return st in ("labeled_original", "labeled_aug") if include_labeled_aug else st == "labeled_original"
        if mode == "test":
            return st in ("labeled_original", "labeled_aug") if include_labeled_aug else st == "labeled_original"
        if mode == "finetune_train":
            return st in ("labeled_original", "labeled_aug") if include_labeled_aug else st == "labeled_original"
        return False

    selected = [it for it in indexed if it.group_id in target_groups and allowed_type(it.sample_type)]

    leaked = [it for it in selected if it.group_id not in target_groups]
    if leaked:
        raise RuntimeError(f"Leakage detected in mode={mode}, fold={fold_id}: {len(leaked)} files")

    forbidden_groups = {
        # 允许 train 与 val 重叠时，训练/验证阶段仅禁止命中 test
        "pretrain": set(split["test"]),
        "finetune_train": set(split["test"]),
        "finetune_val": set(split["test"]),
        # 测试集仍与 train/val 隔离
        "test": set(split["train"]) | set(split["val"]),
    }[mode]
    forbidden_hit = [it for it in selected if it.group_id in forbidden_groups]
    if forbidden_hit:
        ex = ", ".join(sorted({x.filename for x in forbidden_hit})[:5])
        raise RuntimeError(f"Leakage detected: forbidden groups in {mode}: {ex}")

    files = [it.path for it in selected]
    if print_stats:
        _print_mode_stats(mode, fold_id, split, selected, include_labeled_aug)
    if strict and len(files) == 0:
        raise RuntimeError(f"No files collected for mode={mode}, fold={fold_id}, dir={data_dir}")
    return files


def _check_group_disjoint(split: Dict[str, List[int]]) -> None:
    """
    允许 train 与 val 重叠，但不允许与 test 重叠。
    """
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    if tr & te or va & te:
        raise ValueError(
            f"Group overlap detected: train∩test={tr & te}, val∩test={va & te}"
        )


def _print_mode_stats(
    mode: str,
    fold_id: str,
    split: Dict[str, List[int]],
    selected: List[IndexedFile],
    include_labeled_aug: bool,
) -> None:
    print("=" * 80)
    print(f"[FoldSplit] fold={fold_id} mode={mode}")
    print(f"[FoldSplit] train_groups={split['train']}")
    print(f"[FoldSplit] val_groups={split['val']}")
    print(f"[FoldSplit] test_groups={split['test']}")

    by_type: Dict[str, int] = {}
    by_group: Dict[int, int] = {}
    for it in selected:
        by_type[it.sample_type] = by_type.get(it.sample_type, 0) + 1
        by_group[it.group_id] = by_group.get(it.group_id, 0) + 1

    print(f"[FoldSplit] selected_files={len(selected)}")
    print(f"[FoldSplit] by_type={by_type}")
    print(f"[FoldSplit] by_group={dict(sorted(by_group.items()))}")
    if mode == "finetune_train":
        print(f"[FoldSplit] include_labeled_aug={include_labeled_aug}")
    print("[FoldSplit] filename_examples=")
    for it in selected[:5]:
        print(f"  - {it.filename} -> gid={it.group_id}, type={it.sample_type}")
    print("=" * 80)
