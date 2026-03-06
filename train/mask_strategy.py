# weld_seg_project/train/mask_strategy.py 掩码策略
import torch
import numpy as np

class RandomMasker:
    """完全随机的mask策略"""
    def __init__(self, mask_ratio=0.1, seed=None):
        """
        mask_ratio: fraction of points to mask per-sample
        seed: optional RNG seed for reproducibility
        """
        self.mask_ratio = mask_ratio
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def generate_mask(self, curvature, valid_mask=None):
        """Return a boolean mask of shape (B, N, 1) marking points to mask.
        
        Args:
            curvature: (B, N, 1) - 仅用于获取形状，不使用其值
            valid_mask: optional (B, N) boolean mask indicating real (non-padding) points.
        """
        B, N, _ = curvature.shape
        mask = torch.zeros_like(curvature, dtype=torch.bool)
        
        for b in range(B):
            if valid_mask is None:
                valid_idx = torch.arange(N, device=curvature.device)
            else:
                valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
            
            if valid_idx.numel() == 0:
                continue
            
            num_points_to_mask = max(1, int(valid_idx.numel() * self.mask_ratio))
            
            # 随机选择要mask的点
            if num_points_to_mask >= valid_idx.numel():
                chosen_idx = valid_idx
            else:
                perm = torch.randperm(valid_idx.numel(), device=curvature.device)
                chosen_idx = valid_idx[perm[:num_points_to_mask]]
            
            mask[b, chosen_idx, 0] = True
        
        return mask


class HighCurvatureMasker:
    def __init__(self, mask_ratio=0.0005, random_frac=0.1, seed=None):
        """
        mask_ratio: fraction of points to mask per-sample (as before)
        random_frac: fraction of the masked points chosen randomly (rest chosen by top curvature)
        seed: optional RNG seed for reproducibility
        """
        self.mask_ratio = mask_ratio
        self.random_frac = float(random_frac)
        if seed is not None:
            torch.manual_seed(seed)

    def generate_mask(self, curvature, valid_mask=None):
        """Return a boolean mask of shape (B, N, 1) marking points to mask.

        This mixes top-curvature sampling with random sampling to ensure the model
        sees both high-curvature regions and some randomly selected contexts.

        Args:
            curvature: (B, N, 1)
            valid_mask: optional (B, N) boolean mask indicating real (non-padding) points.
        """
        B, N, _ = curvature.shape
        mask = torch.zeros_like(curvature, dtype=torch.bool)
        # per-sample counts depend on number of valid points (not padded length)

        for b in range(B):
            curv = curvature[b, :, 0]
            if valid_mask is None:
                valid_idx = torch.arange(N, device=curv.device)
            else:
                valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            num_points_to_mask = max(1, int(valid_idx.numel() * self.mask_ratio))

            # determine counts
            num_random = int(num_points_to_mask * self.random_frac)
            num_top = num_points_to_mask - num_random
            num_top = max(0, num_top)
            num_random = max(0, num_random)

            chosen = set()
            if num_top > 0:
                # topk: if num_top >= N, just choose all
                if num_top >= valid_idx.numel():
                    top_idx = valid_idx
                else:
                    curv_valid = curv[valid_idx]
                    _, rel = torch.topk(curv_valid, k=num_top)
                    top_idx = valid_idx[rel]
                for i in top_idx.tolist():
                    chosen.add(int(i))

            if num_random > 0:
                # choose from remaining indices
                remaining = [int(i) for i in valid_idx.tolist() if int(i) not in chosen]
                if len(remaining) > 0:
                    num_random = min(num_random, len(remaining))
                    rand_idx = torch.randperm(len(remaining), device=curv.device)[:num_random]
                    for ri in rand_idx.tolist():
                        chosen.add(remaining[ri])

            if len(chosen) > 0:
                idxs = torch.tensor(sorted(chosen), device=curv.device, dtype=torch.long)
                mask[b, idxs, 0] = True

        return mask