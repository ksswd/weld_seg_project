# weld_seg_project/train/mask_strategy.py 掩码策略
import torch

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

    def generate_mask(self, curvature):
        """Return a boolean mask of shape (B, N, 1) marking points to mask.

        This mixes top-curvature sampling with random sampling to ensure the model
        sees both high-curvature regions and some randomly selected contexts.
        """
        B, N, _ = curvature.shape
        mask = torch.zeros_like(curvature, dtype=torch.bool)
        num_points_to_mask = max(1, int(N * self.mask_ratio))

        for b in range(B):
            curv = curvature[b, :, 0]

            # determine counts
            num_random = int(num_points_to_mask * self.random_frac)
            num_top = num_points_to_mask - num_random
            num_top = max(0, num_top)
            num_random = max(0, num_random)

            chosen = set()
            if num_top > 0:
                # topk: if num_top >= N, just choose all
                if num_top >= N:
                    top_idx = torch.arange(N, device=curv.device)
                else:
                    _, top_idx = torch.topk(curv, k=num_top)
                for i in top_idx.tolist():
                    chosen.add(i)

            if num_random > 0:
                # choose from remaining indices
                remaining = [i for i in range(N) if i not in chosen]
                if len(remaining) > 0:
                    num_random = min(num_random, len(remaining))
                    rand_idx = torch.randperm(len(remaining), device=curv.device)[:num_random]
                    for ri in rand_idx.tolist():
                        chosen.add(remaining[ri])

            if len(chosen) > 0:
                idxs = torch.tensor(sorted(chosen), device=curv.device, dtype=torch.long)
                mask[b, idxs, 0] = True

        return mask