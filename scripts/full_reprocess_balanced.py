#!/usr/bin/env python
"""
完整的重新预处理流程：
1. 基于4个标注文件计算归一化参数
2. 重新处理这4个文件（保证归一化一致）
3. 平衡采样（half_v和lap1主导）
"""
import os
import sys
import numpy as np
sys.path.insert(0, '.')

from preprocess.preprocess import PointCloudPreprocessor
from utils.config import Config
from utils.io_utils import read_all_ply_from_dir

print("=" * 80)
print("完整重新预处理：解决归一化不匹配 + 数据平衡")
print("=" * 80)

config = Config()

# 步骤1: 找到4个标注文件对应的原始ply
print("\n步骤1: 查找原始点云文件")
print("-" * 80)

label_files = ['half_v', 'lap1', 'MeshE1_seg', 'T1']
raw_dir = "data/raw/new"

# 创建临时目录存放4个原始文件
temp_raw_dir = "data/raw/label_only"
os.makedirs(temp_raw_dir, exist_ok=True)

import shutil
for name in label_files:
    # 查找对应的ply文件（可能有_aug后缀）
    ply_candidates = [f for f in os.listdir(raw_dir)
                      if f.startswith(name) and f.endswith('.ply')]

    if ply_candidates:
        # 优先选择没有_aug的，或者第一个aug
        ply_file = None
        for candidate in ply_candidates:
            if '_aug_' not in candidate:
                ply_file = candidate
                break

        if ply_file is None:
            # 如果都是aug，选第一个
            ply_file = sorted(ply_candidates)[0]

        src = os.path.join(raw_dir, ply_file)
        dst = os.path.join(temp_raw_dir, f"{name}.ply")
        shutil.copy(src, dst)
        print(f"  ✓ {name}: {ply_file} -> {name}.ply")
    else:
        print(f"  ❌ {name}: 未找到对应的ply文件")

# 步骤2: 基于这4个文件计算归一化参数
print("\n步骤2: 计算归一化参数（仅基于4个标注文件）")
print("-" * 80)

preprocessor = PointCloudPreprocessor(config)
preprocessor.fit(temp_raw_dir)
preprocessor.save_params("normalization_params_labeled_only.npz")

print("  ✓ 归一化参数已保存")

# 步骤3: 重新处理这4个文件
print("\n步骤3: 重新处理标注文件（使用新的归一化参数）")
print("-" * 80)

output_dir = "data/processed/label_reprocessed"
preprocessor.process_and_save_dataset(temp_raw_dir, output_dir)

# 步骤4: 添加标签（从原始label文件复制）
print("\n步骤4: 添加标签信息")
print("-" * 80)

original_label_dir = "data/processed/label"

for name in label_files:
    # 读取重新处理的文件
    reprocessed_file = os.path.join(output_dir, f"{name}.npz")
    if not os.path.exists(reprocessed_file):
        print(f"  ❌ {name}: 重新处理的文件不存在")
        continue

    reprocessed_data = np.load(reprocessed_file)

    # 读取原始标签
    original_file = os.path.join(original_label_dir, f"{name}_label.npz")
    if not os.path.exists(original_file):
        print(f"  ❌ {name}: 原始标签文件不存在")
        continue

    original_data = np.load(original_file)

    # 合并
    merged_data = dict(reprocessed_data)
    if 'labels' in original_data:
        merged_data['labels'] = original_data['labels']

    # 保存
    save_path = os.path.join(output_dir, f"{name}_label.npz")
    np.savez_compressed(save_path, **merged_data)
    print(f"  ✓ {name}: 已添加标签")

# 步骤5: 平衡采样
print("\n步骤5: 平衡采样（half_v和lap1主导）")
print("-" * 80)

output_balanced_dir = "data/processed/label_final"
os.makedirs(output_balanced_dir, exist_ok=True)

# 采样参数
target_main = 10000  # half_v, lap1
target_aux = 5000    # T1, MeshE1

def balanced_sample(data, target_points, name):
    features = data['features']
    labels = data['labels'].squeeze()
    n_total = len(features)

    if n_total <= target_points:
        print(f"  {name}: 保持原样 ({n_total:,}点)")
        return data

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    ratio = n_pos / n_total
    target_pos = int(target_points * ratio)
    target_neg = target_points - target_pos

    sampled_pos = np.random.choice(pos_idx, min(target_pos, n_pos), replace=False)
    sampled_neg = np.random.choice(neg_idx, min(target_neg, n_neg), replace=False)

    sampled_idx = np.concatenate([sampled_pos, sampled_neg])
    np.random.shuffle(sampled_idx)

    new_data = {k: v[sampled_idx] for k, v in data.items()}

    print(f"  {name}: {n_total:,} → {len(sampled_idx):,}点 "
          f"(焊缝: {n_pos:,} → {len(sampled_pos):,})")

    return new_data

# 处理文件
for name in label_files:
    src_file = os.path.join(output_dir, f"{name}_label.npz")
    if not os.path.exists(src_file):
        continue

    data = dict(np.load(src_file))

    # 判断主导还是辅助
    if name in ['half_v', 'lap1']:
        target = target_main
        print(f"\n{name} (主导):")
    else:
        target = target_aux
        print(f"\n{name} (辅助):")

    new_data = balanced_sample(data, target, name)

    # 保存
    save_path = os.path.join(output_balanced_dir, f"{name}_label.npz")
    np.savez_compressed(save_path, **new_data)

# 最终统计
print("\n" + "=" * 80)
print("最终数据统计:")
print("=" * 80)

total = 0
for name in label_files:
    file_path = os.path.join(output_balanced_dir, f"{name}_label.npz")
    if os.path.exists(file_path):
        data = np.load(file_path)
        n = len(data['features'])
        total += n
        n_pos = (data['labels'] == 1).sum()
        print(f"\n{name}:")
        print(f"  点数: {n:,} ({100*n/30000:.1f}%)")
        print(f"  焊缝: {n_pos:,} ({100*n_pos/n:.2f}%)")

print(f"\n总计: {total:,}点")

print("\n" + "=" * 80)
print("✓ 完整重新预处理完成！")
print("\n生成的文件:")
print(f"  1. 归一化参数: normalization_params_labeled_only.npz")
print(f"  2. 平衡后数据: {output_balanced_dir}/")
print("\n下一步:")
print("  1. 更新 config.py:")
print("     LABEL_DATA_DIR = 'data/processed/label_final'")
print("  2. 重新训练:")
print("     python main.py --mode train")
print("\n预期效果:")
print("  ✓ 归一化基于4个文件（不再是160个）")
print("  ✓ half_v和lap1主导（各33%）")
print("  ✓ T1和MeshE1辅助（各17%）")
print("=" * 80)
