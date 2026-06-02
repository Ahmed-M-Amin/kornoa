"""Loss helpers for classifier training."""

from __future__ import annotations

import torch
from torch import nn


class BinaryFocalLoss(nn.Module):
    """Focal loss for one-logit binary classification."""

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("focal loss alpha must be between 0.0 and 1.0")
        if gamma < 0.0:
            raise ValueError("focal loss gamma must be non-negative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("focal loss reduction must be none, mean, or sum")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.reshape(1).to(dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probabilities = torch.sigmoid(logits)
        p_t = (probabilities * targets) + ((1.0 - probabilities) * (1.0 - targets))
        alpha_t = (self.alpha * targets) + ((1.0 - self.alpha) * (1.0 - targets))
        loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


def create_weighted_bce_loss(
    *,
    negative_count: int,
    positive_count: int,
    device: torch.device | str | None = None,
) -> nn.BCEWithLogitsLoss:
    """Create weighted BCE loss using negative/positive class ratio."""

    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("weighted BCE requires positive and negative examples")
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    if device is not None:
        pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def create_binary_focal_loss(
    *,
    negative_count: int | None = None,
    positive_count: int | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    use_pos_weight: bool = False,
    device: torch.device | str | None = None,
) -> BinaryFocalLoss:
    """Create binary focal loss, optionally with the V1 positive-class weight."""

    pos_weight = None
    if use_pos_weight:
        if positive_count is None or negative_count is None or positive_count <= 0 or negative_count <= 0:
            raise ValueError("weighted focal loss requires positive and negative examples")
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
        if device is not None:
            pos_weight = pos_weight.to(device)
    return BinaryFocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
