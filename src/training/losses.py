"""Loss helpers for SPEC-005 classifier training."""

from __future__ import annotations

import torch
from torch import nn


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
