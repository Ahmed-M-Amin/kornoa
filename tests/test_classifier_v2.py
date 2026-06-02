"""Tests for SPEC-007 V2 classifier model and loss support."""

import pytest
import torch

from src.models.classifier import SUPPORTED_CLASSIFIER_BACKBONES, create_classifier
from src.training.losses import BinaryFocalLoss, create_binary_focal_loss


def test_binary_focal_loss_downweights_easy_examples():
    loss = BinaryFocalLoss(alpha=0.5, gamma=2.0, reduction="none")

    easy = loss(torch.tensor([5.0]), torch.tensor([1.0]))
    hard = loss(torch.tensor([-1.0]), torch.tensor([1.0]))

    assert easy.item() < hard.item()


def test_weighted_focal_loss_uses_positive_class_weight():
    loss = create_binary_focal_loss(
        negative_count=8,
        positive_count=2,
        use_pos_weight=True,
    )

    assert torch.isclose(loss.pos_weight, torch.tensor([4.0])).all()


def test_v2_candidate_backbones_are_declared():
    assert {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "convnext_tiny"}.issubset(
        SUPPORTED_CLASSIFIER_BACKBONES
    )


@pytest.mark.parametrize("model_name", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "convnext_tiny"])
def test_v2_candidate_backbones_can_be_requested(model_name):
    pytest.importorskip("torchvision")

    model = create_classifier(model_name=model_name, num_classes=1)

    assert model.model_name == model_name
    assert model.num_classes == 1
