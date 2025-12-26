#!/usr/bin/env python
"""
重新平衡训练数据：让half_v和lap1主导，T1和MeshE1辅助
策略：下采样T1和MeshE1，使4个文件点数相近
"""
import os
import numpy as np
import shutil

print("=" * 80)
print("数据重采样：让half_v和lap1主导训练")
print("=" * 80)

# 配置
label_dir = "data/processed/label"
output_dir = "data/processed/label_balanced"
os.makedirs(output_dir, exist_ok=True)

# 读取所有文件
files = {
    'half_v': 'half_v_label.npz',
    'lap1': 'lap1_label.npz',
    'MeshE1': 'MeshE1_seg_label.npz',
    'T1': 'T1_label.npz'
}

data_dict = {}
for name, fname in files.items():
    path = os.path.join(label_dir, fname)
    data_dict[name] = np.load(path)
    print(f"\n{name}:")
    print(f"  文件: {fname}")
    print(f"  点数: {len(data_dict[name]['features']):,}")
    print(f"  焊缝点: {(data_dict[name]['labels'] == 1).sum():,}")

# 设定目标点数（基于主导文件）
target_main = 10000  # half_v和lap1的目标点数（保持原样或稍微上采样）
target_aux = 5000    # T1和MeshE1的目标点数（辅助）

print("\n" + "=" * 80)
print("重采样策略:")
print("=" * 80)
print(f"  主导文件 (half_v, lap1): 保持原样 (~10,000点)")
print(f"  辅助文件 (T1, MeshE1):   下采样到 {target_aux:,}点")

# 重��样函数
def balanced_sample(data, target_points, name):
    """
    平衡采样：保持正负样本比例
    """
    features = data['features']
    labels = data['labels'].squeeze()

    n_total = len(features)

    if n_total <= target_points:
        print(f"\n  {name}: 保持原样 ({n_total:,}点)")
        return data

    # 分离正负样本
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # 按比例采样
    ratio = n_pos / n_total
    target_pos = int(target_points * ratio)
    target_neg = target_points - target_pos

    # 随机采样
    if n_pos > target_pos:
        sampled_pos = np.random.choice(pos_idx, target_pos, replace=False)
    else:
        sampled_pos = pos_idx

    if n_neg > target_neg:
        sampled_neg = np.random.choice(neg_idx, target_neg, replace=False)
    else:
        sampled_neg = neg_idx

    # 合并索引
    sampled_idx = np.concatenate([sampled_pos, sampled_neg])
    np.random.shuffle(sampled_idx)

    # 创建新数据
    new_data = {}
    for key in data.keys():
        if key == 'labels' or key == 'features':
            new_data[key] = data[key][sampled_idx]
        else:
            new_data[key] = data[key][sampled_idx]

    print(f"\n  {name}: {n_total:,} → {len(sampled_idx):,}点")
    print(f"    正样本: {n_pos:,} → {len(sampled_pos):,}")
    print(f"    负样本: {n_neg:,} → {len(sampled_neg):,}")

    return new_data

# 处理每个文件
print("\n" + "=" * 80)
print("执行重采样:")
print("=" * 80)

# half_v和lap1：主导文件，保持原样
for name in ['half_v', 'lap1']:
    print(f"\n{name} (主导):")
    new_data = balanced_sample(data_dict[name], target_main, name)

    # 保存
    save_path = os.path.join(output_dir, files[name])
    np.savez_compressed(save_path, **new_data)
    print(f"  ✓ 保存到: {save_path}")

# T1和MeshE1：辅助文件，下采样
for name in ['T1', 'MeshE1']:
    print(f"\n{name} (辅助):")
    new_data = balanced_sample(data_dict[name], target_aux, name)

    # 保存
    save_path = os.path.join(output_dir, files[name])
    np.savez_compressed(save_path, **new_data)
    print(f"  ✓ 保存到: {save_path}")

# 统计最终结果
print("\n" + "=" * 80)
print("重采样后统计:")
print("=" * 80)

total_points = 0
total_pos = 0

for name, fname in files.items():
    path = os.path.join(output_dir, fname)
    data = np.load(path)
    n_points = len(data['features'])
    n_pos = (data['labels'] == 1).sum()

    total_points += n_points
    total_pos += n_pos

    print(f"\n{name}:")
    print(f"  点数: {n_points:,} ({100*n_points/total_points:.1f}%)")
    print(f"  焊缝点: {n_pos:,} ({100*n_pos/n_points:.2f}%)")

print(f"\n总计:")
print(f"  总点数: {total_points:,}")
print(f"  焊缝点: {total_pos:,} ({100*total_pos/total_points:.2f}%)")

print("\n" + "=" * 80)
print("✓ 数据重采样完成！")
print("\n下一步:")
print("  1. 更新 config.py: LABEL_DATA_DIR = 'data/processed/label_balanced'")
print("  2. 重新训练: python main.py --mode train")
print("\n预期效果:")
print("  • half_v和lap1主导训练（各占~33%）")
print("  • T1和MeshE1辅助（各占~17%）")
print("  • 4个文件影响力相当")
print("=" * 80)
