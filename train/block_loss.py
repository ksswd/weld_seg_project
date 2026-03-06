# weld_seg_project/train/block_loss.py 块级重建损失
import torch
import torch.nn.functional as F
from typing import List

def block_recon_criterion(recon, batch, mask, blocks_list, block_labels_list):
    """
    块级重建损失（预测曲率）
    
    对每个被mask的块计算loss，然后按块内被mask的点数加权平均
    
    Args:
        recon: (B, N, 1) - 预测的曲率
        batch: 包含curvature的batch字典
        mask: List[torch.Tensor] - 每个样本的点的mask，List长度为B，每个元素是(N,) bool
        blocks_list: List[List[List[int]]] - 每个样本的块列表，List长度为B
        block_labels_list: List[List[str]] - 每个样本的块标签列表，List长度为B
    
    Returns:
        loss: scalar tensor
    """
    curvature = batch['curvature']  # (B, N, 1)
    B, N, _ = recon.shape
    
    # 处理曲率归一化（与pretrain.py中的recon_criterion保持一致）
    from utils.config import Config as GlobalConfig
    curv_mode = str(getattr(GlobalConfig, "PRETRAIN_CURV_TARGET", "log")).lower().strip()
    if curv_mode == "log":
        eps = float(getattr(GlobalConfig, "PRETRAIN_CURV_EPS", 1e-6))
        curv_t = torch.log(curvature.clamp_min(0) + eps)
    else:
        curv_t = curvature
    
    # 收集所有块的loss和权重
    block_losses = []
    block_weights = []
    
    for b in range(B):  # 遍历batch
        sample_blocks = blocks_list[b]
        sample_labels = block_labels_list[b]
        sample_mask = mask[b]  # (N,) bool
        sample_recon = recon[b]  # (N, 1)
        sample_curv = curv_t[b]  # (N, 1)
        
        for block_id, block_indices in enumerate(sample_blocks):
            if len(block_indices) == 0:
                continue
            
            # 检查这个块是否被mask（至少有一个点被mask）
            block_mask = sample_mask[block_indices]
            num_masked = block_mask.sum().item()
            
            if num_masked == 0:
                continue  # 这个块没有被mask，跳过
            
            # 计算这个块的loss
            block_recon = sample_recon[block_indices][block_mask]  # (M, 1)
            block_curv = sample_curv[block_indices][block_mask]  # (M, 1)
            
            # MSE loss for this block
            block_loss = F.mse_loss(block_recon, block_curv, reduction='mean')
            
            # 权重 = 块内被mask的点数
            block_weight = torch.tensor(num_masked, dtype=block_loss.dtype, device=block_loss.device)
            
            block_losses.append(block_loss)
            block_weights.append(block_weight)
    
    # 按点数加权平均
    if len(block_losses) == 0:
        # 如果没有被mask的块，返回0（但这种情况不应该发生）
        return torch.tensor(0.0, device=recon.device, requires_grad=True)
    
    block_losses = torch.stack(block_losses)
    block_weights = torch.stack(block_weights)
    
    # 加权平均
    total_loss = (block_losses * block_weights).sum() / block_weights.sum().clamp_min(1e-8)
    
    return total_loss
