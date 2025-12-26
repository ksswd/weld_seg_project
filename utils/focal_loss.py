# utils/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for positive class (0-1). For imbalanced data, use alpha = neg/(pos+neg)
               For 24:1 ratio, alpha ≈ 0.04 (1/25)
        gamma: Focusing parameter (default=2). Higher gamma means more focus on hard examples.
        reduction: 'none', 'mean', or 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, N) or (B, N, 1) - raw logits from model
            targets: (B, N) or (B, N, 1) - binary labels (0 or 1)
        Returns:
            loss: scalar if reduction='mean'/'sum', else same shape as logits
        """
        # Ensure shapes are (B, N)
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Final focal loss
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
