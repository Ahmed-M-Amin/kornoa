"""Tests for SPEC-007 V1-vs-V2 comparison contracts."""

import json

import pytest

from src.training.metrics import (
    V1BaselineRecord,
    V2CandidateResult,
    generate_v1_vs_v2_comparison,
    load_v1_baseline_record,
    select_v2_candidate,
)


def _v1() -> V1BaselineRecord:
    return V1BaselineRecord(
        artifact_root="artifacts/kaggle_v1_artifacts/outputs/kaggle_v1",
        validation_f1=0.91656,
        kaggle_public_score=0.91693,
        threshold=0.48,
        false_positives=344,
        false_negatives=344,
        uncertain_samples=12,
        average_time_per_image=0.01,
    )


def _candidate(name: str, f1: float, speed: float, *, threshold: float = 0.5) -> V2CandidateResult:
    return V2CandidateResult(
        experiment_name=name,
        backbone="efficientnet_b1",
        image_size=384,
        validation_f1=f1,
        best_threshold=threshold,
        false_positives=300,
        false_negatives=280,
        uncertain_samples=8,
        average_time_per_image=0.01 * speed,
        speed_multiplier_vs_v1=speed,
        model_size_bytes=123,
    )


def test_close_f1_candidates_prefer_faster_model():
    selected = select_v2_candidate([_candidate("slow", 0.94, 1.8), _candidate("fast", 0.9381, 1.1)])

    assert selected.experiment_name == "fast"
    assert selected.selected is True


def test_candidates_over_2x_speed_ceiling_are_not_selectable():
    selected = select_v2_candidate([_candidate("too_slow", 0.96, 2.1), _candidate("fast", 0.94, 1.4)])

    assert selected.experiment_name == "fast"


def test_comparison_report_marks_missing_v2_kaggle_score_unavailable():
    report = generate_v1_vs_v2_comparison(v1=_v1(), candidates=[_candidate("v2b", 0.94, 1.5, threshold=0.52)])

    assert report.validation_f1_delta == pytest.approx(0.02344)
    assert report.threshold_delta == pytest.approx(0.04)
    assert report.false_positive_delta == -44
    assert report.false_negative_delta == -64
    assert report.kaggle_public_score_delta == "unavailable"
    assert report.meets_f1_target is True
    assert report.within_speed_ceiling is True
    assert report.close_f1_tolerance == pytest.approx(0.002)


def test_load_v1_baseline_record_reads_saved_artifacts(tmp_path):
    root = tmp_path / "artifacts" / "kaggle_v1"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "classifier_metrics.json").write_text(
        json.dumps({"f1_score": 0.91, "confusion_counts": {"fp": 10, "fn": 11}}),
        encoding="utf-8",
    )
    (root / "reports" / "best_threshold.json").write_text(json.dumps({"threshold": 0.47}), encoding="utf-8")
    benchmark = tmp_path / "v1_inference_benchmark.json"
    benchmark.write_text(json.dumps({"average_time_per_image": 0.02, "model_name": "efficientnet_b0", "image_size": 384}), encoding="utf-8")
    summary = tmp_path / "hard_example_summary.json"
    summary.write_text(json.dumps({"counts": {"uncertain": 5}}), encoding="utf-8")

    record = load_v1_baseline_record(artifact_root=root, benchmark_path=benchmark, hard_example_summary_path=summary)

    assert record.validation_f1 == pytest.approx(0.91)
    assert record.threshold == pytest.approx(0.47)
    assert record.false_positives == 10
    assert record.false_negatives == 11
    assert record.uncertain_samples == 5
    assert record.average_time_per_image == pytest.approx(0.02)
