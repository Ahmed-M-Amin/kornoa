"""Tests for Spec 018 V2B cleaned replay training contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.training.train_classifier import (
    TrainingValidationError,
    load_classifier_config,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> Path:
    return _write_csv(
        path,
        ["image_id", "true_label", "probability", "threshold", "predicted_label"],
        rows,
    )


def _make_dataset_root(root: Path, *, missing_image: bool = False, drop_label: str | None = None) -> Path:
    dataset_root = root / "dataset"
    train_images = dataset_root / "train_images"
    train_images.mkdir(parents=True, exist_ok=True)
    rows = [
        {"image_id": "keep_000.jpg", "target": 0},
        {"image_id": "keep_001.jpg", "target": 1},
        {"image_id": "keep_002.jpg", "target": 0},
        {"image_id": "keep_003.jpg", "target": 1},
    ]
    if drop_label is not None:
        rows = [row for row in rows if row["image_id"] != drop_label]
    _write_csv(dataset_root / "train.csv", ["image_id", "target"], rows)
    for image_id in ["keep_000.jpg", "keep_001.jpg", "keep_002.jpg", "keep_003.jpg"]:
        if missing_image and image_id == "keep_003.jpg":
            continue
        (train_images / image_id).write_bytes(b"image")
    return dataset_root


def _make_locked_baseline_inputs(root: Path) -> dict[str, Path]:
    checkpoint = root / "baseline" / "models" / "classifier_best.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"baseline")
    predictions = _write_predictions(
        root / "baseline" / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "keep_002.jpg", "true_label": 0, "probability": 0.20, "threshold": 0.50, "predicted_label": 0},
            {"image_id": "keep_003.jpg", "true_label": 1, "probability": 0.80, "threshold": 0.50, "predicted_label": 1},
        ],
    )
    threshold = _write_json(root / "baseline" / "reports" / "best_threshold.json", {"threshold": 0.28, "candidate_count": 17})
    metrics = _write_json(
        root / "baseline" / "reports" / "classifier_metrics.json",
        {
            "f1_score": 0.9269,
            "confusion_counts": {"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            "validation_target_distribution": {"0": 1, "1": 1},
            "test_prediction_distribution": {"0": 5, "1": 7},
        },
    )
    runtime = _write_json(
        root / "baseline" / "benchmarks" / "v2b_inference_benchmark.json",
        {
            "total_time_seconds": 3.0,
            "average_time_per_image": 1.5,
            "images_per_second": 0.666667,
            "batch_size": 1,
            "device": "cpu",
            "image_size": 384,
            "benchmark_reference": "v2b",
        },
    )
    return {
        "checkpoint": checkpoint,
        "predictions": predictions,
        "threshold": threshold,
        "metrics": metrics,
        "runtime": runtime,
    }


def _make_cleaned_replay_config_fixture(
    root: Path,
    *,
    overlap_auto_exclude: bool = False,
    overlap_adjudication: bool = False,
    overlap_deferred: bool = False,
    missing_image: bool = False,
    missing_label: bool = False,
) -> Path:
    baseline_inputs = _make_locked_baseline_inputs(root)
    dataset_root = _make_dataset_root(root, missing_image=missing_image, drop_label="keep_003.jpg" if missing_label else None)
    decision_root = root / "outputs" / "analysis" / "data_quality_decision_lock"
    approved_rows = [
        {"image_id": "keep_000.jpg", "target": 0, "decision_bucket": "keep"},
        {"image_id": "keep_001.jpg", "target": 1, "decision_bucket": "keep"},
        {"image_id": "keep_002.jpg", "target": 0, "decision_bucket": "keep"},
        {"image_id": "keep_003.jpg", "target": 1, "decision_bucket": "keep"},
    ]
    _write_csv(
        decision_root / "approved_cleaned_training_manifest.csv",
        ["image_id", "target", "decision_bucket"],
        approved_rows,
    )
    _write_csv(
        decision_root / "auto_exclude_rows.csv",
        ["image_id"],
        [{"image_id": "keep_000.jpg" if overlap_auto_exclude else "blocked_auto_000.jpg"}],
    )
    _write_csv(
        decision_root / "needs_adjudication_rows.csv",
        ["image_id"],
        [{"image_id": "keep_001.jpg" if overlap_adjudication else "blocked_review_000.jpg"}],
    )
    _write_csv(
        decision_root / "deferred_uncertain_rows.csv",
        ["image_id"],
        [{"image_id": "keep_002.jpg" if overlap_deferred else "blocked_defer_000.jpg"}],
    )
    _write_json(
        decision_root / "reports" / "decision_lock_summary.json",
        {
            "approved_manifest_count": 4,
            "auto_exclude_count": 1,
            "needs_adjudication_count": 1,
            "defer_count": 1,
            "blocked_overlap_count": 0,
        },
    )
    config_path = root / "v2b_cleaned_replay_training.yaml"
    payload = {
        "experiment": {"name": "cleaned_replay_v2b", "seed": 42},
        "classifier": {
            "v2": True,
            "experiment_name": "cleaned_replay_v2b",
            "model_name": "efficientnet_b1",
            "image_size": 384,
            "num_classes": 2,
            "batch_size": 4,
            "epochs": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "imbalance_strategy": "focal_loss_weighted_sampler",
            "augmentation_recipe": "v2_safe",
            "hard_example_strategy": "none",
            "weighted_sampler": True,
            "hard_example_source": "auto",
            "split_source": "v2b_compatible",
        },
        "data": {"dataset_root": str(dataset_root), "split_source": "v2b_compatible"},
        "output": {"root": str(root / "outputs" / "kaggle_v2b_cleaned_replay" / "cleaned_replay_v2b")},
        "benchmark": {"speed_ceiling_multiplier": 2.0},
        "cleaned_replay": {
            "enabled": True,
            "approved_manifest_path": str(decision_root / "approved_cleaned_training_manifest.csv"),
            "auto_exclude_rows_path": str(decision_root / "auto_exclude_rows.csv"),
            "needs_adjudication_rows_path": str(decision_root / "needs_adjudication_rows.csv"),
            "deferred_uncertain_rows_path": str(decision_root / "deferred_uncertain_rows.csv"),
            "decision_lock_summary_path": str(decision_root / "reports" / "decision_lock_summary.json"),
        },
        "baseline": {
            "baseline_name": "v2b_locked_baseline",
            "baseline_checkpoint_path": str(baseline_inputs["checkpoint"]),
            "locked_baseline_predictions_path": str(baseline_inputs["predictions"]),
            "locked_baseline_threshold_path": str(baseline_inputs["threshold"]),
            "locked_baseline_metrics_path": str(baseline_inputs["metrics"]),
            "locked_baseline_runtime_report_path": str(baseline_inputs["runtime"]),
        },
        "safety": {
            "allow_test_labels": False,
            "public_leaderboard_input": False,
            "generate_submission": False,
            "apply_relabels": False,
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _make_smoke_split(train_classifier, tmp_path: Path):
    examples = [
        train_classifier.TrainingExample(image_id="keep_000.jpg", image_path=tmp_path / "keep_000.jpg", label=0),
        train_classifier.TrainingExample(image_id="keep_001.jpg", image_path=tmp_path / "keep_001.jpg", label=1),
        train_classifier.TrainingExample(image_id="keep_002.jpg", image_path=tmp_path / "keep_002.jpg", label=0),
        train_classifier.TrainingExample(image_id="keep_003.jpg", image_path=tmp_path / "keep_003.jpg", label=1),
        train_classifier.TrainingExample(image_id="blocked_auto_000.jpg", image_path=tmp_path / "blocked_auto_000.jpg", label=0),
        train_classifier.TrainingExample(image_id="blocked_review_000.jpg", image_path=tmp_path / "blocked_review_000.jpg", label=1),
    ]
    split = train_classifier.SplitAssignment(train=examples[:2], validation=examples[2:4])
    return examples, split


def test_cleaned_replay_config_scaffold_loads_expected_contract():
    config = load_classifier_config("configs/v2b_cleaned_replay_training.yaml")

    assert config.cleaned_replay_enabled is True
    assert config.v2 is True
    assert config.baseline_name == "v2b_locked_baseline"
    assert config.approved_manifest_path.endswith("approved_cleaned_training_manifest.csv")
    config_text = Path("configs/v2b_cleaned_replay_training.yaml").read_text(encoding="utf-8")
    assert "auto_exclude_rows_path: outputs/analysis/data_quality_decision_lock/auto_exclude_rows.csv" in config_text
    assert "needs_adjudication_rows_path: outputs/analysis/data_quality_decision_lock/needs_adjudication_rows.csv" in config_text
    assert "deferred_uncertain_rows_path: outputs/analysis/data_quality_decision_lock/deferred_uncertain_rows.csv" in config_text


def test_cleaned_replay_dry_run_passes_with_valid_config_and_writes_validation_report(tmp_path):
    from src.training.train_classifier import main

    config_path = _make_cleaned_replay_config_fixture(tmp_path)

    exit_code = main(["--config", str(config_path), "--dry-run-cleaned-replay"])

    report_path = tmp_path / "outputs" / "kaggle_v2b_cleaned_replay" / "cleaned_replay_v2b" / "reports" / "cleaned_replay_dry_run_validation.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["approved_manifest_count"] == 4
    assert payload["auto_exclude_overlap_count"] == 0
    assert payload["needs_adjudication_overlap_count"] == 0
    assert payload["deferred_overlap_count"] == 0
    assert payload["training_started"] is False
    assert payload["no_test_labels_used"] is True
    assert payload["no_submission_created"] is True
    assert payload["no_leaderboard_tuning"] is True


def test_cleaned_replay_dry_run_fails_on_approved_auto_exclude_overlap(tmp_path):
    from src.training.train_classifier import run_cleaned_replay_dry_run_validation

    config_path = _make_cleaned_replay_config_fixture(tmp_path, overlap_auto_exclude=True)

    with pytest.raises(TrainingValidationError, match="auto-excluded overlap"):
        run_cleaned_replay_dry_run_validation(config_path)


def test_cleaned_replay_dry_run_fails_on_approved_needs_adjudication_overlap(tmp_path):
    from src.training.train_classifier import run_cleaned_replay_dry_run_validation

    config_path = _make_cleaned_replay_config_fixture(tmp_path, overlap_adjudication=True)

    with pytest.raises(TrainingValidationError, match="needs-adjudication overlap"):
        run_cleaned_replay_dry_run_validation(config_path)


def test_cleaned_replay_dry_run_fails_on_approved_deferred_overlap(tmp_path):
    from src.training.train_classifier import run_cleaned_replay_dry_run_validation

    config_path = _make_cleaned_replay_config_fixture(tmp_path, overlap_deferred=True)

    with pytest.raises(TrainingValidationError, match="deferred overlap"):
        run_cleaned_replay_dry_run_validation(config_path)


def test_cleaned_replay_dry_run_fails_on_missing_train_image_or_label(tmp_path):
    from src.training.train_classifier import run_cleaned_replay_dry_run_validation

    missing_image_config = _make_cleaned_replay_config_fixture(tmp_path / "missing_image", missing_image=True)
    with pytest.raises(TrainingValidationError, match="missing approved training images"):
        run_cleaned_replay_dry_run_validation(missing_image_config)

    missing_label_config = _make_cleaned_replay_config_fixture(tmp_path / "missing_label", missing_label=True)
    with pytest.raises(TrainingValidationError, match="missing approved training labels"):
        run_cleaned_replay_dry_run_validation(missing_label_config)


def test_cleaned_replay_dry_run_does_not_call_training(tmp_path, monkeypatch):
    import src.training.train_classifier as train_classifier

    config_path = _make_cleaned_replay_config_fixture(tmp_path)

    def _unexpected_run_training(*args, **kwargs):
        raise AssertionError("run_training should not be called during --dry-run-cleaned-replay")

    monkeypatch.setattr(train_classifier, "run_training", _unexpected_run_training)

    exit_code = train_classifier.main(["--config", str(config_path), "--dry-run-cleaned-replay"])

    assert exit_code == 0


def test_cleaned_replay_dry_run_allows_empty_blocked_side_csvs(tmp_path):
    from src.training.train_classifier import run_cleaned_replay_dry_run_validation

    config_path = _make_cleaned_replay_config_fixture(tmp_path)
    decision_root = tmp_path / "outputs" / "analysis" / "data_quality_decision_lock"
    _write_csv(decision_root / "deferred_uncertain_rows.csv", ["image_id"], [])

    report_path = run_cleaned_replay_dry_run_validation(config_path)

    assert report_path.exists()


def test_cleaned_replay_training_filters_examples_to_approved_manifest_rows(tmp_path, monkeypatch):
    import src.training.train_classifier as train_classifier

    config_path = _make_cleaned_replay_config_fixture(tmp_path)
    config = train_classifier.load_classifier_config(config_path)
    examples, split = _make_smoke_split(train_classifier, tmp_path)
    seen_examples: list[object] = []

    monkeypatch.setattr(
        train_classifier,
        "_load_training_examples_with_report",
        lambda dataset_root: train_classifier.TrainingExampleLoadResult(
            examples=examples,
            resolved_dataset_root=Path(config.dataset_root),
            train_csv_row_count=4,
            train_image_count=4,
        ),
    )

    def _capture_split(examples_arg, **kwargs):
        seen_examples.extend(examples_arg)
        return split

    monkeypatch.setattr(train_classifier, "make_stratified_split", _capture_split)
    monkeypatch.setattr(train_classifier, "prepare_hard_example_report", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(train_classifier, "select_training_device", lambda device: (_ for _ in ()).throw(SystemExit(0)))

    with pytest.raises(SystemExit):
        train_classifier.run_training(
            dataset_root=config.dataset_root,
            output_root=config.output_root,
            config=config,
            synthetic_smoke=True,
        )

    assert {example.image_id for example in seen_examples} == {"keep_000.jpg", "keep_001.jpg", "keep_002.jpg", "keep_003.jpg"}


def test_cleaned_replay_training_refuses_to_start_when_dry_run_safety_validation_fails(tmp_path):
    import src.training.train_classifier as train_classifier

    config_path = _make_cleaned_replay_config_fixture(tmp_path, overlap_auto_exclude=True)
    config = train_classifier.load_classifier_config(config_path)

    with pytest.raises(TrainingValidationError, match="auto-excluded overlap"):
        train_classifier.run_training(
            dataset_root=config.dataset_root,
            output_root=config.output_root,
            config=config,
            synthetic_smoke=True,
        )


def test_cleaned_replay_training_smoke_writes_manifest_snapshot_runtime_and_comparison(tmp_path, monkeypatch):
    import src.training.train_classifier as train_classifier

    config_path = _make_cleaned_replay_config_fixture(tmp_path)
    config = train_classifier.load_classifier_config(config_path)
    examples, split = _make_smoke_split(train_classifier, tmp_path)

    monkeypatch.setattr(
        train_classifier,
        "_load_training_examples_with_report",
        lambda dataset_root: train_classifier.TrainingExampleLoadResult(
            examples=examples,
            resolved_dataset_root=Path(config.dataset_root),
            train_csv_row_count=4,
            train_image_count=4,
        ),
    )
    monkeypatch.setattr(train_classifier, "make_stratified_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(
        __import__("src.training.hard_example_mining", fromlist=["prepare_hard_example_report"]),
        "prepare_hard_example_report",
        lambda **kwargs: __import__("src.training.hard_example_mining", fromlist=["HardExampleSourceReport"]).HardExampleSourceReport(
            false_positives_path=tmp_path / "fp.csv",
            false_negatives_path=tmp_path / "fn.csv",
            uncertain_path=tmp_path / "uncertain.csv",
            high_loss_samples_path=tmp_path / "high_loss.csv",
            summary_path=tmp_path / "summary.json",
            loaded_counts={},
            summary_counts={},
            count_mismatches={},
            eligible_for_training_count=0,
            excluded_validation_count=0,
            validation_excluded_image_ids=[],
            used_for_oversampling_count=0,
            train_validation_disjoint=True,
            excluded_rows=[],
            oversampled_image_ids=[],
            oversampled_image_ids_by_group={},
            hard_example_source_used="none",
            hard_example_source_type="none",
        ),
    )
    monkeypatch.setattr(train_classifier, "select_training_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        train_classifier,
        "create_classifier",
        lambda **kwargs: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 1)),
    )
    monkeypatch.setattr(train_classifier, "load_start_checkpoint_if_configured", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_classifier,
        "build_training_dataloaders",
        lambda *args, **kwargs: train_classifier.TrainingDataLoaders(
            train=[(torch.zeros((2, 1, 2, 2)), torch.tensor([[0.0], [1.0]]), ["keep_000.jpg", "keep_001.jpg"])],
            validation=[],
        ),
    )
    monkeypatch.setattr(train_classifier, "_collect_validation_predictions", lambda *args, **kwargs: ([0, 1], [0.2, 0.9]))
    monkeypatch.setattr(
        train_classifier,
        "find_best_threshold",
        lambda **kwargs: train_classifier.ThresholdSearchResult(
            threshold=0.28,
            f1_score=0.95,
            tie_break="max_f1",
            candidate_count=2,
        ),
    )
    monkeypatch.setattr(
        train_classifier,
        "compute_binary_metrics",
        lambda **kwargs: train_classifier.BinaryMetrics(
            f1_score=0.95,
            threshold=0.28,
            confusion_counts={"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            class_counts={"0": 1, "1": 1},
        ),
    )
    monkeypatch.setattr(
        train_classifier,
        "_build_predictions",
        lambda validation_examples, probabilities, threshold: [
            train_classifier.ValidationPrediction(
                image_id="keep_002.jpg",
                true_label=0,
                probability=0.2,
                threshold=threshold,
                predicted_label=0,
                prob_bad=0.2,
                classifier_prediction=0,
                target=0,
            ),
            train_classifier.ValidationPrediction(
                image_id="keep_003.jpg",
                true_label=1,
                probability=0.9,
                threshold=threshold,
                predicted_label=1,
                prob_bad=0.9,
                classifier_prediction=1,
                target=1,
            ),
        ],
    )

    result = train_classifier.run_training(
        dataset_root=config.dataset_root,
        output_root=config.output_root,
        config=config,
        synthetic_smoke=True,
    )

    reports_root = Path(config.output_root) / "reports"
    benchmarks_root = Path(config.output_root) / "benchmarks"
    assert result.metrics_path.exists()
    assert (reports_root / "approved_manifest_snapshot.csv").exists()
    assert (reports_root / "cleaned_replay_run_manifest.json").exists()
    assert (reports_root / "cleaned_replay_comparison.json").exists()
    assert (benchmarks_root / "runtime_metadata.json").exists()
    assert not (Path(config.output_root) / "submissions").exists()


def test_cleaned_replay_comparison_rejects_when_f1_is_below_locked_v2b(tmp_path):
    from src.training.train_classifier import build_cleaned_replay_comparison_record

    baseline_inputs = _make_locked_baseline_inputs(tmp_path)
    candidate_root = tmp_path / "candidate"
    metrics_path = _write_json(
        candidate_root / "reports" / "classifier_metrics.json",
        {
            "f1_score": 0.85,
            "confusion_counts": {"tp": 1, "fp": 1, "fn": 0, "tn": 0},
            "validation_target_distribution": {"0": 1, "1": 1},
        },
    )
    threshold_path = _write_json(candidate_root / "reports" / "best_threshold.json", {"threshold": 0.31})
    predictions_path = _write_predictions(
        candidate_root / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "keep_002.jpg", "true_label": 0, "probability": 0.30, "threshold": 0.31, "predicted_label": 0},
            {"image_id": "keep_003.jpg", "true_label": 1, "probability": 0.80, "threshold": 0.31, "predicted_label": 1},
        ],
    )
    runtime_path = _write_json(candidate_root / "benchmarks" / "runtime_metadata.json", {"runtime_seconds": 12.0})
    dry_run_path = _write_json(candidate_root / "reports" / "cleaned_replay_dry_run_validation.json", {"validation_split_status": "passed", "training_started": False})

    report = build_cleaned_replay_comparison_record(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
        cleaned_metrics_path=metrics_path,
        cleaned_threshold_path=threshold_path,
        cleaned_predictions_path=predictions_path,
        cleaned_runtime_path=runtime_path,
        dry_run_report_path=dry_run_path,
    )

    assert report["decision"] == "rejected"
    assert report["f1_delta"] < 0


def test_cleaned_replay_comparison_accepts_only_when_f1_beats_locked_v2b_and_safety_passes(tmp_path):
    from src.training.train_classifier import build_cleaned_replay_comparison_record

    baseline_inputs = _make_locked_baseline_inputs(tmp_path)
    candidate_root = tmp_path / "candidate"
    metrics_path = _write_json(
        candidate_root / "reports" / "classifier_metrics.json",
        {
            "f1_score": 0.95,
            "confusion_counts": {"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            "validation_target_distribution": {"0": 1, "1": 1},
        },
    )
    threshold_path = _write_json(candidate_root / "reports" / "best_threshold.json", {"threshold": 0.25})
    predictions_path = _write_predictions(
        candidate_root / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "keep_002.jpg", "true_label": 0, "probability": 0.10, "threshold": 0.25, "predicted_label": 0},
            {"image_id": "keep_003.jpg", "true_label": 1, "probability": 0.90, "threshold": 0.25, "predicted_label": 1},
        ],
    )
    runtime_path = _write_json(candidate_root / "benchmarks" / "runtime_metadata.json", {"runtime_seconds": 10.0})
    dry_run_path = _write_json(
        candidate_root / "reports" / "cleaned_replay_dry_run_validation.json",
        {
            "validation_split_status": "passed",
            "training_started": False,
            "no_test_labels_used": True,
            "no_submission_created": True,
            "no_leaderboard_tuning": True,
        },
    )

    report = build_cleaned_replay_comparison_record(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
        cleaned_metrics_path=metrics_path,
        cleaned_threshold_path=threshold_path,
        cleaned_predictions_path=predictions_path,
        cleaned_runtime_path=runtime_path,
        dry_run_report_path=dry_run_path,
    )

    assert report["decision"] == "accepted"
    assert report["f1_delta"] > 0


def test_cleaned_replay_comparison_contains_required_report_fields(tmp_path):
    from src.training.train_classifier import build_cleaned_replay_comparison_record

    baseline_inputs = _make_locked_baseline_inputs(tmp_path)
    candidate_root = tmp_path / "candidate"
    metrics_path = _write_json(
        candidate_root / "reports" / "classifier_metrics.json",
        {
            "f1_score": 0.95,
            "confusion_counts": {"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            "validation_target_distribution": {"0": 1, "1": 1},
        },
    )
    threshold_path = _write_json(candidate_root / "reports" / "best_threshold.json", {"threshold": 0.25})
    predictions_path = _write_predictions(
        candidate_root / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "keep_002.jpg", "true_label": 0, "probability": 0.10, "threshold": 0.25, "predicted_label": 0},
            {"image_id": "keep_003.jpg", "true_label": 1, "probability": 0.90, "threshold": 0.25, "predicted_label": 1},
        ],
    )
    runtime_path = _write_json(candidate_root / "benchmarks" / "runtime_metadata.json", {"runtime_seconds": 10.0})
    dry_run_path = _write_json(
        candidate_root / "reports" / "cleaned_replay_dry_run_validation.json",
        {
            "validation_split_status": "passed",
            "training_started": False,
            "no_test_labels_used": True,
            "no_submission_created": True,
            "no_leaderboard_tuning": True,
        },
    )

    report = build_cleaned_replay_comparison_record(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
        cleaned_metrics_path=metrics_path,
        cleaned_threshold_path=threshold_path,
        cleaned_predictions_path=predictions_path,
        cleaned_runtime_path=runtime_path,
        dry_run_report_path=dry_run_path,
    )

    required = {
        "baseline_name",
        "baseline_validation_f1",
        "cleaned_replay_validation_f1",
        "f1_delta",
        "baseline_threshold",
        "cleaned_replay_threshold",
        "validation_row_count",
        "decision",
        "decision_reason",
        "safety_gate_status",
        "artifact_paths",
    }
    assert required.issubset(report.keys())
