#!/usr/bin/env python
"""
诊断数据增强和归一化的匹配问题
"""
import os
import numpy as np

print("=" * 80)
print("诊断：数据增强和归一化匹配性")
print("=" * 80)

# 1. 检查各目录的文件数量
dirs = {
    "原始数据 (raw)": "data/raw/new",
    "处理后 (processed)": "data/processed/new",
    "标注数据 (label)": "data/processed/label",
    "测试数据 (test)": "data/test/new",
}

print("\n1. 各目录文件统计:")
for name, path in dirs.items():
    if os.path.exists(path):
        ply_count = len([f for f in os.listdir(path) if f.endswith('.ply')])
        npz_count = len([f for f in os.listdir(path) if f.endswith('.npz') and '_pred' not in f])
        print(f"\n   {name}:")
        print(f"     路径: {path}")
        print(f"     .ply: {ply_count}")
        print(f"     .npz: {npz_count}")
    else:
        print(f"\n   {name}: ❌ 不存在 ({path})")

# 2. 检查标注数据的文件名模式
print("\n2. 标注数据文件名分析:")
label_dir = "data/processed/label"
if os.path.exists(label_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.npz') and '_pred' not in f]
    print(f"\n   共 {len(label_files)} 个文件:")
    for f in label_files:
        # 检查是否有增强标记（如 _aug_1）
        if '_aug_' in f:
            print(f"     {f} ⚠️ 包含增强标记")
        else:
            print(f"     {f}")

        # 检查数据量
        data = np.load(os.path.join(label_dir, f))
        n_points = len(data['features'])
        if 'labels' in data:
            n_pos = (data['labels'] == 1).sum()
            print(f"       点数: {n_points:,}, 焊缝点: {n_pos:,} ({100*n_pos/n_points:.2f}%)")

# 3. 检查测试数据的文件名模式
print("\n3. 测试数据文件名分析:")
test_dir = "data/test/new"
if os.path.exists(test_dir):
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
    print(f"\n   共 {len(test_files)} 个文件")

    # 分析文件名前缀
    prefixes = {}
    for f in test_files:
        # 提取基础名称（去掉_aug_数字）
        base = f.split('_aug_')[0] if '_aug_' in f else f.replace('.npz', '')
        if base not in prefixes:
            prefixes[base] = 0
        prefixes[base] += 1

    print(f"\n   文件前缀统计:")
    for prefix, count in sorted(prefixes.items()):
        print(f"     {prefix}: {count} 个变体")

    print(f"\n   不同前缀数: {len(prefixes)}")
    print(f"   总文件数: {sum(prefixes.values())}")

    if len(prefixes) == 16 and sum(prefixes.values()) == 160:
        print("\n   ✓ 确认：16个基础点云 × 10个增强变体 = 160个文件")

# 4. 关键问题诊断
print("\n" + "=" * 80)
print("🔍 关键诊断:")
print("=" * 80)

# 检查训练集是否只是增强数据的子集
label_dir = "data/processed/label"
if os.path.exists(label_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.npz') and '_pred' not in f]
    label_bases = set()
    for f in label_files:
        base = f.replace('_label.npz', '')
        label_bases.add(base)

    print(f"\n问题1: 训练集构成")
    print(f"  标注文件基础名: {label_bases}")
    print(f"  文件数量: {len(label_files)}")

    if len(label_files) == 4 and 'T1' in str(label_bases):
        print("\n  ⚠️  WARNING: 只有4个标注文件，且T1占数据89%")
        print("     这可能导致:")
        print("     - 训练数据严重不足（只有4个样本）")
        print("     - 过拟合到T1的特定模式")
        print("     - 无法泛化到其他数据")

# 5. 归一化参数来源
print(f"\n问题2: 归一化参数来源")
print("  当前流程: 在 data/raw/new/ 上计算归一化参数")

raw_dir = "data/raw/new"
if os.path.exists(raw_dir):
    raw_count = len([f for f in os.listdir(raw_dir) if f.endswith('.ply')])
    print(f"  data/raw/new/ 文件数: {raw_count}")

    if raw_count == 160:
        print("\n  ⚠️  WARNING: 归一化基于160个增强文件")
        print("     但训练只用了4个标注文件")
        print("     这导致:")
        print("     - 归一化参数(mu, sigma)包含了未用于训练的数据")
        print("     - 训练数据的分布可能不匹配归一化参数")
    elif raw_count == 16:
        print("\n  ✓ 归一化基于16个原始文件（合理）")
    else:
        print(f"\n  ⚠️  文件数 {raw_count} 不符合预期（应该是16或160）")

print("\n" + "=" * 80)
print("💡 结论:")
print("=" * 80)

print("""
如果看到以上WARNING，说明存在严重的数据不匹配问题：

【核心矛盾】
  • 归一化参数: 基于160个增强文件（或16个原始文件）
  • 训练数据:   只有4个标注文件
  • 测试数据:   160个增强文件（但归一化参数不匹配训练集）

【导致的后果】
  1. 训练集太小（只有4个样本），严重欠拟合
  2. T1占89%，过拟合到T1的模式
  3. 测试时数据分布和训练时不一致

【解决方案】
  方案A: 增加标注数据（把160个文件都标注）
  方案B: 只在4个已标注的原始文件上训练和测试
  方案C: 移除T1，只用剩余3个文件（避免T1主导）
""")
print("=" * 80)
