"""Metric helpers for SPEC-005 classifier training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    f1_score: float
    threshold: float
    confusion_counts: dict[str, int]
    class_counts: dict[str, int]


def compute_binary_metrics(
    *,
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> BinaryMetrics:
    """Compute F1, confusion counts, and class counts for binary predictions."""

    labels = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if labels.shape != probs.shape:
        raise ValueError("labels and probabilities must have the same length")

    predictions = (probs >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    denominator = (2 * tp) + fp + fn
    f1 = (2 * tp / denominator) if denominator else 0.0

    return BinaryMetrics(
        f1_score=float(f1),
        threshold=float(threshold),
        confusion_counts={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        class_counts={"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
    )
