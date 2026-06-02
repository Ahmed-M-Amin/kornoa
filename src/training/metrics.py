"""Metric helpers for classifier training and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    f1_score: float
    threshold: float
    confusion_counts: dict[str, int]
    class_counts: dict[str, int]


@dataclass(frozen=True)
class V1BaselineRecord:
    artifact_root: str
    validation_f1: float
    threshold: float
    false_positives: int
    false_negatives: int
    uncertain_samples: int
    average_time_per_image: float
    model_name: str = "efficientnet_b0"
    image_size: int = 384
    kaggle_public_score: float | None = None


@dataclass(frozen=True)
class V2CandidateResult:
    experiment_name: str
    backbone: str
    image_size: int
    validation_f1: float
    best_threshold: float
    false_positives: int
    false_negatives: int
    uncertain_samples: int
    average_time_per_image: float
    speed_multiplier_vs_v1: float
    model_size_bytes: int = 0
    selected: bool = False
    selection_reason: str = ""


@dataclass(frozen=True)
class V1VsV2ComparisonReport:
    v1: V1BaselineRecord
    v2_selected: V2CandidateResult
    candidate_results: list[V2CandidateResult]
    validation_f1_delta: float
    threshold_delta: float
    false_positive_delta: int
    false_negative_delta: int
    uncertain_delta: int
    speed_delta: float
    model_size_delta: int | None
    image_size_delta: int
    kaggle_public_score_delta: float | str
    meets_f1_target: bool
    within_speed_ceiling: bool
    speed_ceiling_multiplier: float
    close_f1_tolerance: float
    recommendation: str


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


def select_v2_candidate(
    candidates: Iterable[V2CandidateResult],
    *,
    speed_ceiling_multiplier: float = 2.0,
    close_f1_tolerance: float = 0.002,
) -> V2CandidateResult:
    """Select by validation F1 after threshold search, preferring speed for close F1."""

    eligible = [candidate for candidate in candidates if candidate.speed_multiplier_vs_v1 <= speed_ceiling_multiplier]
    if not eligible:
        raise ValueError("No V2 candidates are within the configured speed ceiling")
    best_f1 = max(candidate.validation_f1 for candidate in eligible)
    close = [
        candidate
        for candidate in eligible
        if abs(best_f1 - candidate.validation_f1) <= close_f1_tolerance
    ]
    selected = min(close, key=lambda candidate: (candidate.speed_multiplier_vs_v1, candidate.average_time_per_image))
    return _mark_selected(selected, "highest_f1_fastest_close_tie")


def generate_v1_vs_v2_comparison(
    *,
    v1: V1BaselineRecord,
    candidates: Iterable[V2CandidateResult],
    speed_ceiling_multiplier: float = 2.0,
    close_f1_tolerance: float = 0.002,
    v2_kaggle_public_score: float | None = None,
) -> V1VsV2ComparisonReport:
    """Build the mandatory SPEC-007 V1-vs-V2 comparison report."""

    candidate_list = list(candidates)
    selected = select_v2_candidate(
        candidate_list,
        speed_ceiling_multiplier=speed_ceiling_multiplier,
        close_f1_tolerance=close_f1_tolerance,
    )
    updated_candidates = [
        selected if candidate.experiment_name == selected.experiment_name else candidate
        for candidate in candidate_list
    ]
    kaggle_delta: float | str = "unavailable"
    if v2_kaggle_public_score is not None and v1.kaggle_public_score is not None:
        kaggle_delta = float(v2_kaggle_public_score - v1.kaggle_public_score)
    f1_delta = float(selected.validation_f1 - v1.validation_f1)
    return V1VsV2ComparisonReport(
        v1=v1,
        v2_selected=selected,
        candidate_results=updated_candidates,
        validation_f1_delta=f1_delta,
        threshold_delta=float(selected.best_threshold - v1.threshold),
        false_positive_delta=int(selected.false_positives - v1.false_positives),
        false_negative_delta=int(selected.false_negatives - v1.false_negatives),
        uncertain_delta=int(selected.uncertain_samples - v1.uncertain_samples),
        speed_delta=float(selected.average_time_per_image - v1.average_time_per_image),
        model_size_delta=None,
        image_size_delta=int(selected.image_size - v1.image_size),
        kaggle_public_score_delta=kaggle_delta,
        meets_f1_target=f1_delta >= 0.02,
        within_speed_ceiling=selected.speed_multiplier_vs_v1 <= speed_ceiling_multiplier,
        speed_ceiling_multiplier=float(speed_ceiling_multiplier),
        close_f1_tolerance=float(close_f1_tolerance),
        recommendation="selected_v2" if f1_delta >= 0.02 else "review_v2_before_replacing_v1",
    )


def load_v1_baseline_record(
    *,
    artifact_root: str | Path,
    benchmark_path: str | Path | None = None,
    hard_example_summary_path: str | Path | None = None,
    kaggle_public_score: float | None = 0.91693,
) -> V1BaselineRecord:
    """Load a V1 baseline record from saved SPEC-005/SPEC-006 artifacts."""

    root = Path(artifact_root)
    root = root / "kaggle_v1" if (root / "kaggle_v1").is_dir() else root
    metrics = _read_json(root / "reports" / "classifier_metrics.json")
    threshold = _read_json(root / "reports" / "best_threshold.json")
    benchmark = _read_json(Path(benchmark_path)) if benchmark_path is not None and Path(benchmark_path).exists() else {}
    summary = (
        _read_json(Path(hard_example_summary_path))
        if hard_example_summary_path is not None and Path(hard_example_summary_path).exists()
        else {}
    )
    confusion = metrics.get("confusion_counts", {})
    counts = summary.get("counts", summary)
    return V1BaselineRecord(
        artifact_root=str(root),
        validation_f1=float(metrics.get("f1_score", 0.91656)),
        threshold=float(threshold.get("threshold", metrics.get("threshold", 0.48))),
        false_positives=int(confusion.get("fp", 344)),
        false_negatives=int(confusion.get("fn", 344)),
        uncertain_samples=int(counts.get("uncertain", 0)),
        average_time_per_image=float(benchmark.get("average_time_per_image", 0.0)),
        model_name=str(benchmark.get("model_name", metrics.get("model_name", "efficientnet_b0"))),
        image_size=int(benchmark.get("image_size", metrics.get("image_size", 384))),
        kaggle_public_score=kaggle_public_score,
    )
def _mark_selected(candidate: V2CandidateResult, reason: str) -> V2CandidateResult:
    return V2CandidateResult(
        experiment_name=candidate.experiment_name,
        backbone=candidate.backbone,
        image_size=candidate.image_size,
        validation_f1=candidate.validation_f1,
        best_threshold=candidate.best_threshold,
        false_positives=candidate.false_positives,
        false_negatives=candidate.false_negatives,
        uncertain_samples=candidate.uncertain_samples,
        average_time_per_image=candidate.average_time_per_image,
        speed_multiplier_vs_v1=candidate.speed_multiplier_vs_v1,
        model_size_bytes=candidate.model_size_bytes,
        selected=True,
        selection_reason=reason,
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
