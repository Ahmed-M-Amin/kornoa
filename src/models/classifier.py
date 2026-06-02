"""Classifier model factory for SPEC-005 binary training."""

from __future__ import annotations

import torch
from torch import nn


class BinaryClassifier(nn.Module):
    """Small wrapper exposing model metadata and one-logit output."""

    def __init__(self, backbone: nn.Module, *, model_name: str, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(inputs)
        return logits.reshape(logits.shape[0])


class TinyCnnBackbone(nn.Module):
    """Fast local model used by synthetic smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return self.classifier(features)


def create_classifier(
    *,
    model_name: str = "efficientnet_b0",
    num_classes: int = 1,
    synthetic_smoke: bool = False,
) -> BinaryClassifier:
    """Create a one-logit binary classifier.

    The default contract is EfficientNet-B0. Synthetic smoke runs can request a
    tiny local backbone to keep automated tests fast and offline-safe.
    """

    if num_classes != 1:
        raise ValueError("SPEC-005 classifier factory expects one binary logit")

    if synthetic_smoke or model_name == "tiny_cnn":
        return BinaryClassifier(TinyCnnBackbone(), model_name=model_name, num_classes=num_classes)

    if model_name != "efficientnet_b0":
        raise ValueError(f"Unsupported classifier model: {model_name}")

    try:
        from torchvision.models import efficientnet_b0
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        raise RuntimeError("torchvision EfficientNet-B0 is unavailable") from exc

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return BinaryClassifier(model, model_name=model_name, num_classes=num_classes)
