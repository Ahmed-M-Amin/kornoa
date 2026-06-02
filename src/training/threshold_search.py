"""Threshold search helpers for SPEC-005 classifier training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.training.metrics import compute_binary_metrics


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    f1_score: float
    tie_break: str
    candidate_count: int


def find_best_threshold(
    *,
    y_true: Iterable[int],
    probabilities: Iterable[float],
    candidates: Sequence[float] | None = None,
) -> ThresholdSearchResult:
    """Find the best F1 threshold, selecting the lowest threshold on ties."""

    labels = list(y_true)
    probs = list(probabilities)
    if candidates is None:
        candidates = [round(index / 100, 2) for index in range(0, 101)]
    if not candidates:
        raise ValueError("threshold search requires at least one candidate")

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for candidate in sorted(float(item) for item in candidates):
        metrics = compute_binary_metrics(y_true=labels, probabilities=probs, threshold=candidate)
        if metrics.f1_score > best_f1:
            best_f1 = metrics.f1_score
            best_threshold = candidate

    return ThresholdSearchResult(
        threshold=best_threshold,
        f1_score=float(best_f1),
        tie_break="lowest_threshold",
        candidate_count=len(candidates),
    )
