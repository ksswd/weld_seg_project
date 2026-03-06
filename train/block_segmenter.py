# weld_seg_project/train/block_segmenter.py 块分割器
import torch
import numpy as np
from typing import List, Tuple

class BlockSegmenter:
    """
    将点云分割成固定大小的网格块，并根据块内高曲率点数量分类
    """
    def __init__(self, 
                 target_points_per_block=1000,  # 每个块的目标点数
                 high_curv_threshold=0.01,  # 高曲率阈值
                 min_high_curv_points=5,  # 高曲率点的最小数量
                 align_grid=True,  # 是否对齐网格（使边界明显）
                 grid_align_base=0.001):  # 网格对齐基数（米）
        """
        Args:
            target_points_per_block: 每个块的目标点数
            high_curv_threshold: 高曲率阈值
            min_high_curv_points: 高曲率点的最小数量（用于分类）
            align_grid: 是否对齐网格，使块边界明显
            grid_align_base: 网格对齐基数，块大小会对齐到这个值的倍数
        """
        self.target_points_per_block = target_points_per_block
        self.high_curv_threshold = high_curv_threshold
        self.min_high_curv_points = min_high_curv_points
        self.align_grid = align_grid
        self.grid_align_base = grid_align_base
    
    def segment(self, points: np.ndarray, curvature: np.ndarray) -> Tuple[List[List[int]], List[str]]:
        """
        将点云分割成块并分类
        
        Args:
            points: (N, 3) - 点坐标
            curvature: (N, 1) or (N,) - 曲率
        
        Returns:
            blocks: List[List[int]] - 每个块是点的索引列表
            block_labels: List[str] - 每个块的标签 ('weld' or 'background')
        """
        if curvature.ndim == 2:
            curvature = curvature.squeeze(1)
        
        N = len(points)
        if N == 0:
            return [], []
        
        # 计算网格大小（自适应）
        block_size = self._compute_block_size(points, N)
        
        # 网格划分
        blocks = self._grid_segment(points, block_size)
        
        # 过滤空块
        blocks = [block for block in blocks if len(block) > 0]
        
        # 分类每个块
        block_labels = []
        for block_indices in blocks:
            if len(block_indices) == 0:
                continue
            block_curv = curvature[block_indices]
            label = self._classify_block(block_curv)
            block_labels.append(label)
        
        return blocks, block_labels
    
    def _compute_block_size(self, points: np.ndarray, num_points: int) -> float:
        """
        根据目标点数计算块大小
        
        Args:
            points: (N, 3) - 点坐标
            num_points: 点的总数
        
        Returns:
            block_size: 块的大小（米）
        """
        # 计算点云的范围
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        ranges = max_coords - min_coords
        
        # 计算每个维度的平均范围（用于估算）
        avg_range = ranges.mean()
        
        if avg_range < 1e-8:
            # 如果范围太小，使用默认值
            return self.grid_align_base if self.align_grid else 0.01
        
        # 估算需要的块数量
        num_blocks = max(1, num_points / self.target_points_per_block)
        
        # 估算每个维度的网格数量（假设是立方体网格）
        grid_count_per_dim = np.ceil(np.power(num_blocks, 1.0/3.0))
        grid_count_per_dim = max(1, int(grid_count_per_dim))
        
        # 计算块大小
        block_size = avg_range / grid_count_per_dim
        
        # 如果启用网格对齐，将块大小对齐到grid_align_base的倍数
        if self.align_grid:
            # 向上取整到grid_align_base的倍数
            block_size = np.ceil(block_size / self.grid_align_base) * self.grid_align_base
            # 确保至少是grid_align_base
            block_size = max(block_size, self.grid_align_base)
        
        return float(block_size)
    
    def _grid_segment(self, points: np.ndarray, block_size: float) -> List[List[int]]:
        """
        网格划分
        
        Args:
            points: (N, 3) - 点坐标
            block_size: 块的大小（米）
        
        Returns:
            blocks: List[List[int]] - 每个块是点的索引列表
        """
        # 计算网格范围
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        # 如果启用网格对齐，将min_coords向下对齐，max_coords向上对齐
        if self.align_grid:
            min_coords = np.floor(min_coords / block_size) * block_size
            max_coords = np.ceil(max_coords / block_size) * block_size
        
        # 计算网格数量（每个维度）
        grid_size = (max_coords - min_coords) / block_size
        grid_size = np.ceil(grid_size).astype(int)
        
        # 确保至少是1x1x1
        grid_size = np.maximum(grid_size, 1)
        
        # 将点分配到网格
        # 计算每个点所在的网格索引
        grid_indices = (points - min_coords) / block_size
        grid_indices = np.floor(grid_indices).astype(int)
        
        # 限制在有效范围内
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        # 构建块（使用字典存储，key是网格坐标）
        blocks_dict = {}
        for i, (x, y, z) in enumerate(grid_indices):
            key = tuple([int(x), int(y), int(z)])
            if key not in blocks_dict:
                blocks_dict[key] = []
            blocks_dict[key].append(i)
        
        # 转换为列表（保持顺序）
        blocks = list(blocks_dict.values())
        
        return blocks
    
    def _classify_block(self, block_curvature: np.ndarray) -> str:
        """
        根据块内高曲率点数量分类块
        
        Args:
            block_curvature: 块内点的曲率
        
        Returns:
            'weld' or 'background'
        """
        high_curv_mask = block_curvature > self.high_curv_threshold
        num_high_curv = high_curv_mask.sum()
        
        if num_high_curv >= self.min_high_curv_points:
            return 'weld'  # 高曲率块（焊缝块）
        else:
            return 'background'  # 背景块
