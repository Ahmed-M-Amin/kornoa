"""Tests for SPEC-015 controlled Phase 3 training contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import yaml
import torch

from src.training.train_classifier import (
    CandidateHypothesisRecord,
    LockedBaselineReport,
    TrainingValidationError,
    aggregate_runtime_evidence,
    build_candidate_evaluation_record,
    build_governed_phase3_training_manifest,
    build_insight_evidence_pack,
    load_classifier_config,
    load_locked_baseline_report,
    validate_baseline_lock_targets,
    validate_candidate_artifact_package,
    validate_candidate_hypothesis,
    write_candidate_comparison_table,
    write_locked_baseline_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "true_label", "probability", "threshold", "predicted_label"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_candidate_package(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "image_id",
            "target",
            "approved_candidate_action",
            "phase3_use_allowed",
            "source_candidate_package_path",
            "baseline_membership_context",
            "hard_example_group",
            "blocked",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_simple_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_dataset_root(root: Path) -> Path:
    dataset_root = root / "dataset"
    (dataset_root / "train_images").mkdir(parents=True, exist_ok=True)
    _write_simple_csv(
        dataset_root / "train.csv",
        ["image_id", "target"],
        [{"image_id": "train_000.jpg", "target": 0}, {"image_id": "train_001.jpg", "target": 1}],
    )
    return dataset_root


def _make_phase3_config_fixture(root: Path, *, overlap_blocked: bool = False, include_source_candidate_package_path: bool = True, blank_source_candidate_package_path: bool = False) -> Path:
    baseline_inputs = _make_locked_baseline_inputs(root)
    dataset_root = _make_dataset_root(root)
    reports_root = root / "outputs" / "analysis" / "hard_row_visual_review" / "reports"
    def _allowed_row(image_id: str, target: int) -> dict[str, object]:
        row: dict[str, object] = {
            "image_id": image_id,
            "target": target,
            "approved_candidate_action": "train",
            "phase3_use_allowed": "true",
            "baseline_membership_context": "baseline_fn",
            "hard_example_group": "false_negative",
            "blocked": "false",
        }
        if include_source_candidate_package_path:
            row["source_candidate_package_path"] = "" if blank_source_candidate_package_path else "spec014/phase3_allowed_hard_training_rows.csv"
        return row
    allowed_rows = [_allowed_row(image_id, target) for image_id, target in [("a.jpg", 1), ("b.jpg", 0), ("c.jpg", 1)]] + [
        _allowed_row(f"allowed_{index:03d}.jpg", index % 2) for index in range(419)
    ]
    _write_candidate_package(reports_root / "phase3_allowed_hard_training_rows.csv", allowed_rows)
    blocked_image_id = "allowed_000.jpg" if overlap_blocked else "blocked_000.jpg"
    _write_simple_csv(
        reports_root / "blocked_roi_pipeline_bug_rows.csv",
        ["image_id"],
        [{"image_id": blocked_image_id}],
    )
    _write_simple_csv(
        reports_root / "blocked_suspected_mislabel_rows.csv",
        ["image_id"],
        [{"image_id": "blocked_001.jpg"}],
    )
    _write_simple_csv(
        reports_root / "blocked_other_high_risk_rows.csv",
        ["image_id"],
        [{"image_id": "blocked_002.jpg"}],
    )
    config_path = root / "phase3_controlled_training.yaml"
    config_payload = {
        "experiment": {"name": "phase3_controlled_candidate_a", "seed": 42},
        "classifier": {
            "v2": True,
            "experiment_name": "phase3_controlled_candidate_a",
            "model_name": "efficientnet_b1",
            "image_size": 384,
            "num_classes": 2,
            "batch_size": 4,
            "epochs": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "imbalance_strategy": "focal_loss_weighted_sampler",
            "augmentation_recipe": "v2_safe",
            "hard_example_strategy": "oversample",
            "weighted_sampler": True,
            "hard_example_source": "auto",
            "split_source": "v2b_compatible",
        },
        "data": {
            "dataset_root": str(dataset_root),
            "split_source": "v2b_compatible",
        },
        "output": {"root": str(root / "outputs" / "kaggle_phase3" / "controlled_phase3")},
        "benchmark": {"speed_ceiling_multiplier": 2.0},
        "hard_examples": {"strategy": "oversample"},
        "phase3_controlled": {
            "enabled": True,
            "baseline_name": "v2b_locked_baseline",
            "baseline_checkpoint_path": str(baseline_inputs["checkpoint"]),
            "locked_baseline_predictions_path": str(baseline_inputs["predictions"]),
            "locked_baseline_threshold_path": str(baseline_inputs["threshold"]),
            "locked_baseline_metrics_path": str(baseline_inputs["metrics"]),
            "locked_baseline_runtime_report_path": str(baseline_inputs["runtime"]),
            "approved_candidate_package_path": str(reports_root / "phase3_allowed_hard_training_rows.csv"),
            "blocked_roi_pipeline_bug_rows_path": str(reports_root / "blocked_roi_pipeline_bug_rows.csv"),
            "blocked_suspected_mislabel_rows_path": str(reports_root / "blocked_suspected_mislabel_rows.csv"),
            "blocked_other_high_risk_rows_path": str(reports_root / "blocked_other_high_risk_rows.csv"),
            "candidate_hypothesis": "Controlled Phase 3 dry-run validation only.",
            "candidate_family": "efficientnet_b1",
            "intended_tradeoff": "Validate governed inputs before training.",
        },
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    return config_path


def _make_locked_baseline_inputs(root: Path) -> dict[str, Path]:
    checkpoint = root / "baseline" / "models" / "classifier_best.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"baseline")
    predictions = _write_predictions(
        root / "baseline" / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "a.jpg", "true_label": 1, "probability": 0.30, "threshold": 0.50, "predicted_label": 0},
            {"image_id": "b.jpg", "true_label": 0, "probability": 0.70, "threshold": 0.50, "predicted_label": 1},
            {"image_id": "c.jpg", "true_label": 1, "probability": 0.90, "threshold": 0.50, "predicted_label": 1},
        ],
    )
    threshold = _write_json(root / "baseline" / "reports" / "best_threshold.json", {"threshold": 0.5, "candidate_count": 9})
    metrics = _write_json(
        root / "baseline" / "reports" / "classifier_metrics.json",
        {
            "f1_score": 0.66,
            "confusion_counts": {"tp": 1, "fp": 1, "fn": 1, "tn": 0},
            "validation_target_distribution": {"0": 1, "1": 2},
            "test_prediction_distribution": {"0": 5, "1": 7},
        },
    )
    runtime = _write_json(
        root / "baseline" / "benchmarks" / "runtime_report.json",
        {
            "total_time_seconds": 3.0,
            "average_time_per_image": 1.0,
            "images_per_second": 1.0,
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


def _make_candidate_outputs(root: Path) -> dict[str, Path]:
    return {
        "validation_predictions": _write_predictions(
            root / "predictions" / "val_predictions.csv",
            [
                {"image_id": "a.jpg", "true_label": 1, "probability": 0.80, "threshold": 0.50, "predicted_label": 1},
                {"image_id": "b.jpg", "true_label": 0, "probability": 0.60, "threshold": 0.50, "predicted_label": 1},
                {"image_id": "c.jpg", "true_label": 1, "probability": 0.90, "threshold": 0.50, "predicted_label": 1},
            ],
        ),
        "test_probabilities": _write_predictions(
            root / "predictions" / "test_probabilities.csv",
            [
                {"image_id": "t1.jpg", "true_label": 0, "probability": 0.20, "threshold": 0.50, "predicted_label": 0},
            ],
        ),
        "best_threshold": _write_json(
            root / "reports" / "best_threshold.json",
            {"threshold": 0.5, "candidate_count": 11, "selection_source": "validation_only"},
        ),
        "metrics": _write_json(
            root / "reports" / "classifier_metrics.json",
            {
                "f1_score": 0.8,
                "confusion_counts": {"tp": 2, "fp": 1, "fn": 0, "tn": 0},
                "validation_target_distribution": {"0": 1, "1": 2},
            },
        ),
        "runtime_report": _write_json(
            root / "benchmarks" / "runtime_report.json",
            {
                "total_time_seconds": 2.0,
                "average_time_per_image": 0.666667,
                "images_per_second": 1.5,
                "batch_size": 1,
                "device": "cpu",
                "image_size": 384,
                "benchmark_reference": "v2b",
            },
        ),
        "target_distribution": _write_json(root / "reports" / "target_distribution_report.json", {"0": 1, "1": 2}),
        "submission": _write_json(root / "submissions" / "submission_phase3.csv", {"rows": 1}),
    }


def test_phase3_controlled_config_scaffold_loads_expected_contract():
    config = load_classifier_config("configs/phase3_controlled_training.yaml")

    assert config.controlled_phase3 is True
    assert config.v2 is True
    assert config.baseline_name == "v2b_locked_baseline"
    assert config.approved_candidate_package_path.endswith("phase3_allowed_hard_training_rows.csv")
    config_text = Path("configs/phase3_controlled_training.yaml").read_text(encoding="utf-8")
    assert "blocked_roi_pipeline_bug_rows_path: outputs/analysis/hard_row_visual_review/reports/blocked_roi_pipeline_bug_rows.csv" in config_text
    assert "blocked_suspected_mislabel_rows_path: outputs/analysis/hard_row_visual_review/reports/blocked_suspected_mislabel_rows.csv" in config_text
    assert "blocked_other_high_risk_rows_path: outputs/analysis/hard_row_visual_review/reports/blocked_other_high_risk_rows.csv" in config_text
    assert config.candidate_hypothesis


def test_locked_baseline_report_loads_and_serializes(tmp_path):
    baseline_inputs = _make_locked_baseline_inputs(tmp_path)

    report = load_locked_baseline_report(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
    )

    output_path = write_locked_baseline_report(report, tmp_path / "outputs" / "analysis" / "phase3_controlled_training" / "locked_baseline_report.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["baseline_name"] == "v2b_locked_baseline"
    assert saved["validation_f1"] == pytest.approx(0.66)
    assert saved["runtime_reference"]["benchmark_reference"] == "v2b"


def test_governed_phase3_manifest_keeps_only_allowed_and_traceable_rows(tmp_path):
    package = _write_candidate_package(
        tmp_path / "candidate_package.csv",
        [
            {
                "image_id": "keep.jpg",
                "target": 1,
                "approved_candidate_action": "train",
                "phase3_use_allowed": "true",
                "source_candidate_package_path": "spec014/candidate_package.csv",
                "baseline_membership_context": "baseline_fn",
                "hard_example_group": "false_negative",
                "blocked": "false",
            },
            {
                "image_id": "blocked.jpg",
                "target": 0,
                "approved_candidate_action": "blocked",
                "phase3_use_allowed": "true",
                "source_candidate_package_path": "spec014/candidate_package.csv",
                "baseline_membership_context": "baseline_fp",
                "hard_example_group": "false_positive",
                "blocked": "true",
            },
            {
                "image_id": "skip.jpg",
                "target": 1,
                "approved_candidate_action": "review",
                "phase3_use_allowed": "false",
                "source_candidate_package_path": "spec014/candidate_package.csv",
                "baseline_membership_context": "",
                "hard_example_group": "uncertain",
                "blocked": "false",
            },
        ],
    )

    manifest = build_governed_phase3_training_manifest(package, output_path=tmp_path / "governed.csv")

    assert [row.image_id for row in manifest] == ["keep.jpg"]
    persisted = list(csv.DictReader((tmp_path / "governed.csv").open("r", newline="", encoding="utf-8")))
    assert persisted[0]["source_candidate_package_path"] == "spec014/candidate_package.csv"


def test_governed_phase3_manifest_fills_missing_traceability_from_configured_candidate_package_path(tmp_path):
    package = _write_candidate_package(
        tmp_path / "candidate_package.csv",
        [
            {
                "image_id": "bad.jpg",
                "target": 1,
                "approved_candidate_action": "train",
                "phase3_use_allowed": "true",
                "baseline_membership_context": "",
                "hard_example_group": "false_negative",
                "blocked": "false",
            }
        ],
    )

    manifest = build_governed_phase3_training_manifest(
        package,
        output_path=tmp_path / "governed.csv",
        default_source_candidate_package_path="outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv",
    )

    assert manifest[0].source_candidate_package_path.endswith("phase3_allowed_hard_training_rows.csv")
    persisted = list(csv.DictReader((tmp_path / "governed.csv").open("r", newline="", encoding="utf-8")))
    assert persisted[0]["source_candidate_package_path"].endswith("phase3_allowed_hard_training_rows.csv")


def test_governed_phase3_manifest_preserves_existing_valid_traceability(tmp_path):
    package = _write_candidate_package(
        tmp_path / "candidate_package.csv",
        [
            {
                "image_id": "keep.jpg",
                "target": 1,
                "approved_candidate_action": "train",
                "phase3_use_allowed": "true",
                "source_candidate_package_path": "spec014/original.csv",
                "baseline_membership_context": "",
                "hard_example_group": "false_negative",
                "blocked": "false",
            }
        ],
    )

    manifest = build_governed_phase3_training_manifest(
        package,
        default_source_candidate_package_path="outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv",
    )

    assert manifest[0].source_candidate_package_path == "spec014/original.csv"


def test_governed_phase3_manifest_fills_blank_traceability_from_configured_candidate_package_path(tmp_path):
    package = _write_candidate_package(
        tmp_path / "candidate_package.csv",
        [
            {
                "image_id": "blank.jpg",
                "target": 1,
                "approved_candidate_action": "train",
                "phase3_use_allowed": "true",
                "source_candidate_package_path": "",
                "baseline_membership_context": "",
                "hard_example_group": "false_negative",
                "blocked": "false",
            }
        ],
    )

    manifest = build_governed_phase3_training_manifest(
        package,
        default_source_candidate_package_path="outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv",
    )

    assert manifest[0].source_candidate_package_path.endswith("phase3_allowed_hard_training_rows.csv")


def test_baseline_lock_rejects_candidate_root_that_overlaps_locked_artifacts(tmp_path):
    baseline_inputs = _make_locked_baseline_inputs(tmp_path)
    report = load_locked_baseline_report(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
    )

    with pytest.raises(TrainingValidationError, match="overwrite"):
        validate_baseline_lock_targets(report, candidate_output_root=baseline_inputs["checkpoint"].parent)


def test_candidate_hypothesis_requires_explicit_statement():
    with pytest.raises(TrainingValidationError, match="explicit hypothesis"):
        validate_candidate_hypothesis(
            CandidateHypothesisRecord(
                candidate_name="phase3_a",
                hypothesis="",
                candidate_family="effnet_b1",
                image_size=384,
                hard_example_strategy="oversample",
                baseline_reference="v2b_locked_baseline",
                intended_tradeoff="higher recall",
            )
        )


def test_candidate_evaluation_record_includes_changed_rows_and_subset_metrics(tmp_path):
    baseline_inputs = _make_locked_baseline_inputs(tmp_path)
    candidate_outputs = _make_candidate_outputs(tmp_path / "candidate")
    baseline_report = load_locked_baseline_report(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
    )

    evaluation = build_candidate_evaluation_record(
        candidate_name="phase3_candidate_a",
        baseline_report=baseline_report,
        metrics_path=candidate_outputs["metrics"],
        validation_predictions_path=candidate_outputs["validation_predictions"],
        test_probabilities_path=candidate_outputs["test_probabilities"],
        best_threshold_path=candidate_outputs["best_threshold"],
        runtime_report_path=candidate_outputs["runtime_report"],
        target_distribution_path=candidate_outputs["target_distribution"],
        submission_path=candidate_outputs["submission"],
        hard_example_image_ids=["a.jpg", "b.jpg"],
        ablation_status="stability_check_passed",
        decision="accepted",
        decision_reason="Hard-example subset improved without new false negatives.",
    )

    assert evaluation.hard_example_f1 > 0
    assert evaluation.non_hard_example_f1 > 0
    assert [row["image_id"] for row in evaluation.changed_rows_vs_baseline] == ["a.jpg"]
    assert evaluation.artifact_completeness_status == "complete"
    assert evaluation.finalist_ready is True


def test_candidate_artifact_package_marks_missing_runtime_as_incomplete(tmp_path):
    candidate_outputs = _make_candidate_outputs(tmp_path / "candidate")
    candidate_outputs["runtime_report"].unlink()
    status = validate_candidate_artifact_package({key: str(value) for key, value in candidate_outputs.items()})

    assert status.startswith("incomplete:")
    assert "runtime_report" in status


def test_comparison_table_rejects_duplicate_candidates_and_persists_rows(tmp_path):
    baseline_report = LockedBaselineReport(
        baseline_name="baseline",
        baseline_checkpoint_path="a",
        validation_predictions_path="b",
        best_threshold_path="c",
        validation_f1=0.5,
        precision=0.5,
        recall=0.5,
        tp=1,
        fp=1,
        fn=1,
        tn=1,
        validation_target_distribution={"0": 1, "1": 1},
        test_target_distribution={"0": 1, "1": 1},
        runtime_reference={},
    )
    candidate_outputs = _make_candidate_outputs(tmp_path / "candidate")
    baseline_inputs = _make_locked_baseline_inputs(tmp_path / "baseline")
    loaded_baseline = load_locked_baseline_report(
        baseline_name=baseline_report.baseline_name,
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
    )
    evaluation = build_candidate_evaluation_record(
        candidate_name="phase3_candidate_a",
        baseline_report=loaded_baseline,
        metrics_path=candidate_outputs["metrics"],
        validation_predictions_path=candidate_outputs["validation_predictions"],
        test_probabilities_path=candidate_outputs["test_probabilities"],
        best_threshold_path=candidate_outputs["best_threshold"],
        runtime_report_path=candidate_outputs["runtime_report"],
        target_distribution_path=candidate_outputs["target_distribution"],
        submission_path=candidate_outputs["submission"],
        hard_example_image_ids=["a.jpg"],
        ablation_status="stability_check_passed",
        decision="accepted",
        decision_reason="Candidate meets the controlled gate.",
    )
    output_path = tmp_path / "reports" / "comparison.csv"

    write_candidate_comparison_table([evaluation], output_path)
    rows = list(csv.DictReader(output_path.open("r", newline="", encoding="utf-8")))
    assert rows[0]["candidate_name"] == "phase3_candidate_a"

    with pytest.raises(TrainingValidationError, match="exactly once"):
        write_candidate_comparison_table([evaluation, evaluation], output_path)


def test_runtime_evidence_and_insight_pack_capture_finalist_decision(tmp_path):
    baseline_inputs = _make_locked_baseline_inputs(tmp_path)
    candidate_outputs = _make_candidate_outputs(tmp_path / "candidate")
    baseline_report = load_locked_baseline_report(
        baseline_name="v2b_locked_baseline",
        baseline_checkpoint_path=baseline_inputs["checkpoint"],
        validation_predictions_path=baseline_inputs["predictions"],
        best_threshold_path=baseline_inputs["threshold"],
        metrics_path=baseline_inputs["metrics"],
        runtime_report_path=baseline_inputs["runtime"],
    )
    evaluation = build_candidate_evaluation_record(
        candidate_name="phase3_candidate_a",
        baseline_report=baseline_report,
        metrics_path=candidate_outputs["metrics"],
        validation_predictions_path=candidate_outputs["validation_predictions"],
        test_probabilities_path=candidate_outputs["test_probabilities"],
        best_threshold_path=candidate_outputs["best_threshold"],
        runtime_report_path=candidate_outputs["runtime_report"],
        target_distribution_path=candidate_outputs["target_distribution"],
        submission_path=candidate_outputs["submission"],
        hard_example_image_ids=["a.jpg", "b.jpg"],
        ablation_status="stability_check_passed",
        decision="accepted",
        decision_reason="Best hard-example tradeoff.",
    )
    second_runtime = _write_json(
        tmp_path / "candidate" / "benchmarks" / "runtime_report_2.json",
        {
            "total_time_seconds": 2.4,
            "average_time_per_image": 0.8,
            "images_per_second": 1.25,
            "batch_size": 1,
            "device": "cpu",
            "image_size": 384,
            "benchmark_reference": "v2b",
        },
    )

    runtime_pack = aggregate_runtime_evidence(
        "phase3_candidate_a",
        [candidate_outputs["runtime_report"], second_runtime],
    )
    insight = build_insight_evidence_pack(
        baseline_report=baseline_report,
        evaluations=[evaluation],
        runtime_pack=runtime_pack,
        final_recommendation_candidate="phase3_candidate_a",
    )

    assert runtime_pack.timing_run_count == 2
    assert runtime_pack.best_runtime_seconds == pytest.approx(2.0)
    assert insight.final_recommendation["candidate_name"] == "phase3_candidate_a"
    assert insight.candidate_outcomes[0]["decision"] == "accepted"


def test_phase3_dry_run_passes_with_valid_config_and_writes_validation_report(tmp_path):
    from src.training.train_classifier import main

    config_path = _make_phase3_config_fixture(tmp_path)

    exit_code = main(["--config", str(config_path), "--dry-run-phase3"])

    report_path = tmp_path / "outputs" / "kaggle_phase3" / "controlled_phase3" / "reports" / "phase3_dry_run_validation.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["allowed_hard_row_count"] == 422
    assert payload["blocked_overlap_count"] == 0
    assert payload["used_test_labels"] is False
    assert payload["submission_enabled"] is False
    assert payload["leaderboard_tuning_enabled"] is False


def test_phase3_dry_run_fails_on_allowed_blocked_overlap(tmp_path):
    from src.training.train_classifier import run_phase3_dry_run_validation

    config_path = _make_phase3_config_fixture(tmp_path, overlap_blocked=True)

    with pytest.raises(TrainingValidationError, match="overlap"):
        run_phase3_dry_run_validation(config_path)


def test_phase3_dry_run_does_not_call_training(tmp_path, monkeypatch):
    import src.training.train_classifier as train_classifier

    config_path = _make_phase3_config_fixture(tmp_path)

    def _unexpected_run_training(*args, **kwargs):
        raise AssertionError("run_training should not be called during --dry-run-phase3")

    monkeypatch.setattr(train_classifier, "run_training", _unexpected_run_training)

    exit_code = train_classifier.main(["--config", str(config_path), "--dry-run-phase3"])

    assert exit_code == 0


def test_phase3_training_run_emits_post_training_report_package_for_smoke_limited_runs(tmp_path, monkeypatch):
    import src.training.train_classifier as train_classifier

    config_path = _make_phase3_config_fixture(tmp_path, include_source_candidate_package_path=False)
    config = train_classifier.load_classifier_config(config_path)

    examples = [
        train_classifier.TrainingExample(image_id="a.jpg", image_path=tmp_path / "a.jpg", label=1),
        train_classifier.TrainingExample(image_id="b.jpg", image_path=tmp_path / "b.jpg", label=0),
        train_classifier.TrainingExample(image_id="c.jpg", image_path=tmp_path / "c.jpg", label=1),
        train_classifier.TrainingExample(image_id="d.jpg", image_path=tmp_path / "d.jpg", label=0),
    ]
    split = train_classifier.SplitAssignment(train=examples[:2], validation=examples[2:])

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
    monkeypatch.setattr(train_classifier, "exclude_hard_examples_from_validation", lambda split, **kwargs: split)
    monkeypatch.setattr(
        train_classifier,
        "select_training_device",
        lambda device: torch.device("cpu"),
    )
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
            train=[(torch.zeros((2, 1, 2, 2)), torch.tensor([[1.0], [0.0]]), ["a.jpg", "b.jpg"])],
            validation=[],
        ),
    )
    monkeypatch.setattr(
        train_classifier,
        "_collect_validation_predictions",
        lambda *args, **kwargs: ([1, 0], [0.9, 0.2]),
    )
    monkeypatch.setattr(
        train_classifier,
        "find_best_threshold",
        lambda **kwargs: train_classifier.ThresholdSearchResult(
            threshold=0.5,
            f1_score=1.0,
            tie_break="max_f1",
            candidate_count=2,
        ),
    )
    monkeypatch.setattr(
        train_classifier,
        "compute_binary_metrics",
        lambda **kwargs: train_classifier.BinaryMetrics(
            f1_score=1.0,
            threshold=0.5,
            confusion_counts={"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            class_counts={"0": 1, "1": 1},
        ),
    )
    monkeypatch.setattr(
        train_classifier,
        "_build_predictions",
        lambda validation_examples, probabilities, threshold: [
            train_classifier.ValidationPrediction(
                image_id="a.jpg",
                true_label=1,
                probability=0.9,
                threshold=threshold,
                predicted_label=1,
                prob_bad=0.9,
                classifier_prediction=1,
                target=1,
            ),
            train_classifier.ValidationPrediction(
                image_id="b.jpg",
                true_label=0,
                probability=0.2,
                threshold=threshold,
                predicted_label=0,
                prob_bad=0.2,
                classifier_prediction=0,
                target=0,
            ),
        ],
    )
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
            eligible_for_training_count=1,
            excluded_validation_count=0,
            validation_excluded_image_ids=[],
            used_for_oversampling_count=1,
            train_validation_disjoint=True,
            excluded_rows=[],
            oversampled_image_ids=["a.jpg"],
            oversampled_image_ids_by_group={"false_negatives": ["a.jpg"]},
            hard_example_source_used="spec014",
            hard_example_source_type="spec014_allowed",
        ),
    )

    result = train_classifier.run_training(
        dataset_root=config.dataset_root,
        output_root=config.output_root,
        config=config,
        synthetic_smoke=True,
    )

    reports_root = Path(config.output_root) / "reports"
    assert result.metrics_path.exists()
    assert (reports_root / "phase3_governed_training_manifest.csv").exists()
    assert (reports_root / "phase3_baseline_lock.json").exists()
    assert (reports_root / "phase3_run_manifest.json").exists()
    assert (reports_root / "phase3_clean_vs_hard_metrics.json").exists()
    assert (reports_root / "phase3_changed_row_evidence.csv").exists()
    assert (reports_root / "phase3_comparison_table.csv").exists()
    assert (reports_root / "phase3_runtime_summary.json").exists()
    assert (reports_root / "phase3_insight_pack.json").exists()
    assert not (Path(config.output_root) / "submissions").exists()
    governed_rows = list(csv.DictReader((reports_root / "phase3_governed_training_manifest.csv").open("r", newline="", encoding="utf-8")))
    assert governed_rows[0]["source_candidate_package_path"] == config.approved_candidate_package_path
