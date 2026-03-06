#!/usr/bin/env python3
# weld_seg_project/scripts/visualize_blocks.py 可视化块分割结果
import os
import sys
import argparse
import numpy as np
import open3d as o3d

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.block_segmenter import BlockSegmenter
from utils.io_utils import load_features_from_csv
from utils.config import Config

def visualize_blocks(points, blocks, block_labels, save_path):
    """
    用不同颜色标注每个块，输出点云
    
    Args:
        points: (N, 3) - 点坐标
        blocks: List[List[int]] - 每个块是点的索引列表
        block_labels: List[str] - 每个块的标签
        save_path: 保存路径
    """
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    # 为每个块分配颜色
    colors = np.zeros((len(points), 3), dtype=np.float64)
    
    # 定义颜色映射
    weld_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # 红色 - 焊缝块
    background_color = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # 蓝色 - 背景块
    
    # 为每个块着色（使用不同的颜色区分不同块）
    import matplotlib.cm as cm
    num_blocks = len(blocks)
    if num_blocks > 0:
        # 使用colormap为每个块分配不同颜色（但保持焊缝/背景的区分）
        colormap = cm.get_cmap('tab20')
        
        for block_id, block_indices in enumerate(blocks):
            if len(block_indices) == 0:
                continue
            
            if block_labels[block_id] == 'weld':
                # 焊缝块：使用红色系（不同深浅）
                base_color = weld_color
                # 添加一些变化以区分不同块
                variation = (block_id % 5) * 0.1
                block_color = np.clip(base_color + variation * np.array([0, 0.2, 0.2]), 0, 1)
            else:
                # 背景块：使用蓝色系（不同深浅）
                base_color = background_color
                # 添加一些变化以区分不同块
                variation = (block_id % 5) * 0.1
                block_color = np.clip(base_color + variation * np.array([0.2, 0.2, 0]), 0, 1)
            
            colors[block_indices] = block_color
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved block visualization to {save_path}")
    
    # 统计信息
    num_weld_blocks = sum(1 for label in block_labels if label == 'weld')
    num_bg_blocks = sum(1 for label in block_labels if label == 'background')
    print(f"Total blocks: {len(blocks)}")
    print(f"  - Weld blocks: {num_weld_blocks}")
    print(f"  - Background blocks: {num_bg_blocks}")
    
    # 统计每个块的点数
    block_sizes = [len(block) for block in blocks]
    if block_sizes:
        print(f"Block size statistics:")
        print(f"  - Min: {min(block_sizes)}")
        print(f"  - Max: {max(block_sizes)}")
        print(f"  - Mean: {np.mean(block_sizes):.1f}")
        print(f"  - Median: {np.median(block_sizes):.1f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize block segmentation')
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--out_dir', type=str, default='data/predictions/block_vis', 
                       help='Output directory for visualization')
    parser.add_argument('--target_points_per_block', type=int, default=1000,
                       help='Target number of points per block')
    parser.add_argument('--high_curv_threshold', type=float, default=0.01,
                       help='High curvature threshold')
    parser.add_argument('--min_high_curv_points', type=int, default=5,
                       help='Minimum number of high curvature points to classify as weld block')
    parser.add_argument('--grid_align_base', type=float, default=0.001,
                       help='Grid alignment base (meters)')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"Loading data from {args.csv_path}...")
    data = load_features_from_csv(args.csv_path)
    points = data['coordinate']
    curvature = data['curvature']
    
    if curvature.ndim == 2 and curvature.shape[1] == 1:
        curvature = curvature.squeeze(1)
    
    print(f"Loaded {len(points)} points")
    
    # 块分割
    print("Segmenting into blocks...")
    segmenter = BlockSegmenter(
        target_points_per_block=args.target_points_per_block,
        high_curv_threshold=args.high_curv_threshold,
        min_high_curv_points=args.min_high_curv_points,
        align_grid=True,
        grid_align_base=args.grid_align_base
    )
    blocks, block_labels = segmenter.segment(points, curvature)
    
    # 可视化
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    save_path = os.path.join(args.out_dir, f'{base_name}_blocks.ply')
    
    visualize_blocks(points, blocks, block_labels, save_path)
    
    print(f"\nVisualization saved to: {save_path}")

if __name__ == '__main__':
    main()
