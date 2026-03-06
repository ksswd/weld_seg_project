# weld_seg_project/train/block_masker.py 块级掩码生成器
import torch
import numpy as np
import random
from typing import List

class BlockMasker:
    """
    块级掩码生成器，支持多种mask策略
    """
    def __init__(self, mask_ratio=0.3, strategy='mixed', seed=None, 
                 weld_mask_ratio=None, bg_mask_ratio=None):
        """
        Args:
            mask_ratio: mask的块比例（0-1），当strategy='mixed'时，这是总比例
            strategy: mask策略 
                - 'mixed': 同时mask两种块（推荐，每个epoch都学习两种任务）
                - 'alternate': 交替mask（偶数epoch mask焊缝块，奇数epoch mask背景块）
                - 'random': 随机mask块
                - 'weld_first': 优先mask焊缝块
            weld_mask_ratio: 焊缝块的mask比例（仅当strategy='mixed'时使用）
            bg_mask_ratio: 背景块的mask比例（仅当strategy='mixed'时使用）
            seed: 随机种子
        """
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        # 如果指定了weld_mask_ratio和bg_mask_ratio，使用它们；否则使用mask_ratio的一半
        self.weld_mask_ratio = weld_mask_ratio if weld_mask_ratio is not None else mask_ratio * 0.5
        self.bg_mask_ratio = bg_mask_ratio if bg_mask_ratio is not None else mask_ratio * 0.5
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_mask(self, blocks: List[List[int]], block_labels: List[str], 
                     epoch: int = None) -> torch.Tensor:
        """
        生成块级mask
        
        Args:
            blocks: List[List[int]] - 每个块是点的索引列表
            block_labels: List[str] - 每个块的标签 ('weld' or 'background')
            epoch: 当前epoch（用于交替策略）
        
        Returns:
            mask: (N,) bool - 点的mask（True表示被mask）
        """
        if len(blocks) == 0:
            return torch.zeros(0, dtype=torch.bool)
        
        # 计算总点数
        N = max(max(block) for block in blocks if len(block) > 0) + 1
        
        mask = torch.zeros(N, dtype=torch.bool)
        
        if self.strategy == 'mixed':
            # 混合策略：同时mask两种块，让模型每个epoch都学习两种任务
            weld_blocks = [i for i, label in enumerate(block_labels) if label == 'weld']
            bg_blocks = [i for i, label in enumerate(block_labels) if label == 'background']
            
            # Mask焊缝块
            if len(weld_blocks) > 0:
                num_weld_to_mask = max(1, int(len(weld_blocks) * self.weld_mask_ratio))
                num_weld_to_mask = min(num_weld_to_mask, len(weld_blocks))
                masked_weld_ids = random.sample(weld_blocks, num_weld_to_mask)
                for block_id in masked_weld_ids:
                    for point_idx in blocks[block_id]:
                        if point_idx < N:
                            mask[point_idx] = True
            
            # Mask背景块
            if len(bg_blocks) > 0:
                num_bg_to_mask = max(1, int(len(bg_blocks) * self.bg_mask_ratio))
                num_bg_to_mask = min(num_bg_to_mask, len(bg_blocks))
                masked_bg_ids = random.sample(bg_blocks, num_bg_to_mask)
                for block_id in masked_bg_ids:
                    for point_idx in blocks[block_id]:
                        if point_idx < N:
                            mask[point_idx] = True
        
        elif self.strategy == 'alternate':
            if epoch is None:
                epoch = 0
            
            if epoch % 2 == 0:
                # 偶数epoch：mask焊缝块
                target_blocks = [i for i, label in enumerate(block_labels) if label == 'weld']
            else:
                # 奇数epoch：mask背景块
                target_blocks = [i for i, label in enumerate(block_labels) if label == 'background']
            
            if len(target_blocks) == 0:
                # 如果没有目标块，随机mask一些块
                target_blocks = list(range(len(blocks)))
            
            # 随机选择要mask的块
            num_blocks_to_mask = max(1, int(len(target_blocks) * self.mask_ratio))
            num_blocks_to_mask = min(num_blocks_to_mask, len(target_blocks))
            masked_block_ids = random.sample(target_blocks, num_blocks_to_mask)
            
            # 生成点的mask
            for block_id in masked_block_ids:
                for point_idx in blocks[block_id]:
                    if point_idx < N:
                        mask[point_idx] = True
        
        elif self.strategy == 'random':
            # 随机mask块
            num_blocks_to_mask = max(1, int(len(blocks) * self.mask_ratio))
            num_blocks_to_mask = min(num_blocks_to_mask, len(blocks))
            masked_block_ids = random.sample(range(len(blocks)), num_blocks_to_mask)
            
            for block_id in masked_block_ids:
                for point_idx in blocks[block_id]:
                    if point_idx < N:
                        mask[point_idx] = True
        
        elif self.strategy == 'weld_first':
            # 优先mask焊缝块
            weld_blocks = [i for i, label in enumerate(block_labels) if label == 'weld']
            bg_blocks = [i for i, label in enumerate(block_labels) if label == 'background']
            
            # 先mask焊缝块
            num_weld_to_mask = max(1, int(len(weld_blocks) * self.mask_ratio))
            num_weld_to_mask = min(num_weld_to_mask, len(weld_blocks))
            masked_weld_ids = random.sample(weld_blocks, num_weld_to_mask) if weld_blocks else []
            
            # 再mask一些背景块
            num_bg_to_mask = max(1, int(len(bg_blocks) * self.mask_ratio * 0.5))
            num_bg_to_mask = min(num_bg_to_mask, len(bg_blocks))
            masked_bg_ids = random.sample(bg_blocks, num_bg_to_mask) if bg_blocks else []
            
            for block_id in masked_weld_ids + masked_bg_ids:
                for point_idx in blocks[block_id]:
                    if point_idx < N:
                        mask[point_idx] = True
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return mask
