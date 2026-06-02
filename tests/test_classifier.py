"""Tests for SPEC-005 classifier model and loss helpers."""

import torch
import pytest

from src.models.classifier import create_classifier
from src.training.losses import create_weighted_bce_loss


def test_classifier_factory_defaults_to_efficientnet_b0_config():
    model = create_classifier(model_name="efficientnet_b0", num_classes=1, synthetic_smoke=True)

    assert model.model_name == "efficientnet_b0"
    assert model.num_classes == 1


def test_classifier_factory_instantiates_real_efficientnet_when_available():
    pytest.importorskip("torchvision")

    model = create_classifier(model_name="efficientnet_b0", num_classes=1, synthetic_smoke=False)

    assert model.model_name == "efficientnet_b0"
    assert model.num_classes == 1
    assert type(model.backbone).__name__ == "EfficientNet"


def test_classifier_forward_outputs_single_binary_logit_per_image():
    model = create_classifier(model_name="tiny_cnn", num_classes=1, synthetic_smoke=True)
    batch = torch.zeros((4, 3, 384, 384), dtype=torch.float32)

    logits = model(batch)
    probabilities = torch.sigmoid(logits)

    assert logits.shape == (4,)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_weighted_bce_loss_uses_positive_class_weight():
    loss = create_weighted_bce_loss(negative_count=8, positive_count=2)

    assert torch.isclose(loss.pos_weight, torch.tensor([4.0])).all()
