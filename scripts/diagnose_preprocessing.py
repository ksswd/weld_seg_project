#!/usr/bin/env python
"""
检查预处理是否破坏了点云几何结构
"""
import numpy as np
import sys
sys.path.insert(0, '.')

print("=" * 80)
print("诊断: 预处理是否破坏点云结构")
print("=" * 80)

# 1. 检查归一化参数
print("\n1. 检查归一化参数 (normalization_params.npz):")
try:
    params = np.load("normalization_params.npz")
    mu = params['mu']
    sigma = params['sigma']

    print(f"\n   mu (mean):")
    print(f"   xyz:      [{mu[0]:.4f}, {mu[1]:.4f}, {mu[2]:.4f}]")
    print(f"   normals:  [{mu[3]:.4f}, {mu[4]:.4f}, {mu[5]:.4f}]")
    print(f"   curv/dens:[{mu[6]:.4f}, {mu[7]:.4f}]")

    print(f"\n   sigma (std):")
    print(f"   xyz:      [{sigma[0]:.4f}, {sigma[1]:.4f}, {sigma[2]:.4f}]")
    print(f"   normals:  [{sigma[3]:.4f}, {sigma[4]:.4f}, {sigma[5]:.4f}]")
    print(f"   curv/dens:[{sigma[6]:.4f}, {sigma[7]:.4f}]")

    # 检查xyz是否被过度归一化
    xyz_range_orig = sigma[:3] * 6  # 假设6-sigma覆盖99.7%
    print(f"\n   原始xyz范围估计: {xyz_range_orig}")

except FileNotFoundError:
    print("   ❌ normalization_params.npz not found!")

# 2. 对比原始ply和处理后的npz
import os
print("\n2. 对比原始点云和处理后点云:")

# 找一个原始ply文件
raw_dir = "data/raw/new"
processed_dir = "data/processed/label"

if os.path.exists(raw_dir):
    ply_files = [f for f in os.listdir(raw_dir) if f.endswith('.ply')]
    if ply_files:
        from utils.io_utils import read_all_ply_from_dir

        ply_file = ply_files[0]
        print(f"\n   原始文件: {ply_file}")

        # 读取原始点云
        point_clouds, _ = read_all_ply_from_dir(raw_dir)
        if point_clouds:
            orig_pc = point_clouds[0]
            print(f"   原始点云形状: {orig_pc.shape}")
            print(f"   原始xyz范围:")
            print(f"     X: [{orig_pc[:, 0].min():.4f}, {orig_pc[:, 0].max():.4f}]")
            print(f"     Y: [{orig_pc[:, 1].min():.4f}, {orig_pc[:, 1].max():.4f}]")
            print(f"     Z: [{orig_pc[:, 2].min():.4f}, {orig_pc[:, 2].max():.4f}]")

            # 读取处理后的
            npz_file = ply_file.replace('.ply', '_label.npz')
            npz_path = os.path.join(processed_dir, npz_file)

            if os.path.exists(npz_path):
                npz_data = np.load(npz_path)
                processed_feat = npz_data['features']

                # 提取归一化后的xyz
                norm_xyz = processed_feat[:, :3]

                print(f"\n   处理后文件: {npz_file}")
                print(f"   归一化后xyz范围:")
                print(f"     X: [{norm_xyz[:, 0].min():.4f}, {norm_xyz[:, 0].max():.4f}]")
                print(f"     Y: [{norm_xyz[:, 1].min():.4f}, {norm_xyz[:, 1].max():.4f}]")
                print(f"     Z: [{norm_xyz[:, 2].min():.4f}, {norm_xyz[:, 2].max():.4f}]")

                # 反归一化
                denorm_xyz = norm_xyz * sigma[:3] + mu[:3]
                print(f"\n   反归一化后xyz范围:")
                print(f"     X: [{denorm_xyz[:, 0].min():.4f}, {denorm_xyz[:, 0].max():.4f}]")
                print(f"     Y: [{denorm_xyz[:, 1].min():.4f}, {denorm_xyz[:, 1].max():.4f}]")
                print(f"     Z: [{denorm_xyz[:, 2].min():.4f}, {denorm_xyz[:, 2].max():.4f}]")

                # 检查误差
                diff = np.abs(denorm_xyz - orig_pc).max()
                print(f"\n   反归一化误差: {diff:.6f}")

                if diff > 0.01:
                    print("   ⚠️  WARNING: 反归一化后和原始点云差异较大！")
                else:
                    print("   ✓ 反归一化正确，xyz坐标应该没问题")

print("\n" + "=" * 80)
print("\n💡 诊断建议:")
print("\n如果看到:")
print("  1. 归一化后xyz范围在[-3, 3]左右 → 正常")
print("  2. 反归一化误差 < 0.01 → xyz坐标没问题")
print("  3. 如果可视化还是歪的 → 可能是可视化代码的问题")
print("\n如果归一化后xyz范围很大(>10) → 归一化参数可能有问题")
print("=" * 80)
