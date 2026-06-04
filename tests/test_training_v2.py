"""Tests for SPEC-007 V2 training contracts."""

from pathlib import Path
import json

import pytest

from src.training.hard_example_mining import prepare_hard_example_report, resolve_hard_example_sources
from src.training.train_classifier import (
    V2_MODEL_OUTPUT,
    V2_METRICS_OUTPUT,
    V2_PREDICTIONS_OUTPUT,
    V2_THRESHOLD_OUTPUT,
    SplitAssignment,
    TrainingExample,
    TrainingRunConfig,
    TrainingValidationError,
    build_hard_example_weight_map,
    build_weighted_random_sampler,
    exclude_hard_examples_from_validation,
    load_classifier_config,
    report_split_disjointness,
    validate_generated_artifact_confidentiality,
    validate_v2_scope_guards,
    _resolve_output_path,
)


def _example(image_id: str, label: int) -> TrainingExample:
    return TrainingExample(image_id=image_id, image_path=Path(f"{image_id}.jpg"), label=label)


def _write_hard_example_csv(path: Path, image_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("image_id,true_label,probability,threshold,predicted_label\n" + "".join(f"{i},0,0.9,0.5,1\n" for i in image_ids), encoding="utf-8")


def test_v2_config_defaults_to_analysis_only_and_safe_recipe():
    config = load_classifier_config("configs/classifier_v2.yaml")

    assert config.v2 is True
    assert config.hard_example_strategy == "analysis_only"
    assert config.hard_example_source == "auto"
    assert config.imbalance_strategy == "focal_loss_weighted_sampler"
    assert config.weighted_sampler is True
    assert config.augmentation_recipe == "v2_safe"
    assert config.speed_ceiling_multiplier == pytest.approx(2.0)
    assert config.close_f1_tolerance == pytest.approx(0.002)


def test_v2_allowed_hard_example_strategies_are_valid():
    from src.training.train_classifier import _validate_training_config

    for strategy in ("none", "analysis_only", "oversample"):
        config = TrainingRunConfig(v2=True, hard_example_strategy=strategy)

        _validate_training_config(config)

    with pytest.raises(TrainingValidationError, match="none, analysis_only, or oversample"):
        _validate_training_config(TrainingRunConfig(v2=True, hard_example_strategy="auto"))


def test_v2_scope_guards_reject_out_of_scope_features():
    with pytest.raises(TrainingValidationError, match="detector"):
        validate_v2_scope_guards({"detector": True})


def test_v2_448_experiment_must_be_explicit():
    from src.training.train_classifier import _validate_training_config

    with pytest.raises(TrainingValidationError, match="448x448"):
        _validate_training_config(TrainingRunConfig(v2=True, image_size=448, experiment_name="v2b_effnet_b1"))


def test_weighted_sampler_gives_minority_class_larger_weight():
    examples = [_example("neg_1", 0), _example("neg_2", 0), _example("neg_3", 0), _example("pos_1", 1)]

    sampler = build_weighted_random_sampler(examples, seed=7)

    weights = sampler.weights.tolist()
    assert weights[-1] > weights[0]


def test_hard_example_oversampling_only_when_explicit(tmp_path):
    _write_hard_example_csv(tmp_path / "hard" / "false_positives.csv", ["train_a", "val_a", "missing_a"])

    analysis = prepare_hard_example_report(
        hard_examples_root=tmp_path / "hard",
        train_image_ids=["train_a"],
        validation_image_ids=["val_a"],
        strategy="analysis_only",
    )
    oversample = prepare_hard_example_report(
        hard_examples_root=tmp_path / "hard",
        train_image_ids=["train_a"],
        validation_image_ids=["val_a"],
        strategy="oversample",
    )

    assert analysis.used_for_oversampling_count == 0
    assert oversample.used_for_oversampling_count == 1
    assert oversample.excluded_validation_count == 1
    assert "val_a" in oversample.excluded_rows
    assert "missing_a" in oversample.excluded_rows
    assert build_hard_example_weight_map(analysis, strategy="analysis_only") == {}
    assert build_hard_example_weight_map(oversample, strategy="oversample") == {"train_a": 2.0}


def test_explicit_hard_example_source_path_is_preferred_over_auto(tmp_path):
    explicit = tmp_path / "explicit_source" / "hard_examples"
    auto = tmp_path / "artifacts" / "kaggle_v2b_artifacts" / "hard_examples"
    _write_hard_example_csv(explicit / "false_negatives.csv", ["explicit_id"])
    _write_hard_example_csv(auto / "false_negatives.csv", ["auto_id"])

    resolved = resolve_hard_example_sources(
        hard_example_source=explicit.parent,
        search_roots=[tmp_path],
    )
    report = prepare_hard_example_report(
        hard_example_source=explicit.parent,
        train_image_ids=["explicit_id", "auto_id"],
        validation_image_ids=[],
        strategy="oversample",
    )

    assert resolved.hard_examples_root == explicit
    assert report.oversampled_image_ids == ["explicit_id"]
    assert report.hard_example_source_used == str(explicit)


def test_explicit_v2b_artifact_source_can_be_used(tmp_path):
    source = tmp_path / "artifacts" / "kaggle_v2b_artifacts"
    hard_root = source / "hard_examples"
    _write_hard_example_csv(hard_root / "false_negatives.csv", ["v2b_fn"])

    report = prepare_hard_example_report(
        hard_example_source=source,
        train_image_ids=["v2b_fn"],
        validation_image_ids=[],
        strategy="oversample",
    )

    assert report.hard_example_source_used == str(hard_root)
    assert report.loaded_counts["false_negatives"] == 1
    assert report.used_for_oversampling_count == 1


def test_explicit_source_generates_hard_examples_from_predictions_and_threshold(tmp_path):
    source = tmp_path / "artifacts" / "kaggle_v2b_artifacts"
    predictions = source / "kaggle_v2" / "predictions" / "val_classifier_predictions.csv"
    threshold = source / "kaggle_v2" / "reports" / "best_threshold.json"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    threshold.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(
        "image_id,true_label,probability,threshold,predicted_label\n"
        "fn_id,1,0.10,0.50,0\n"
        "fp_id,0,0.90,0.50,1\n"
        "ok_id,1,0.80,0.50,1\n",
        encoding="utf-8",
    )
    threshold.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

    resolved = resolve_hard_example_sources(hard_example_source=source)
    report = prepare_hard_example_report(
        hard_example_source=source,
        train_image_ids=["fn_id", "fp_id"],
        validation_image_ids=[],
        strategy="oversample",
    )

    assert resolved.generated is True
    assert (source / "hard_examples" / "false_negatives.csv").exists()
    assert (source / "hard_examples" / "false_positives.csv").exists()
    assert (source / "hard_examples" / "uncertain.csv").exists()
    assert (source / "hard_examples" / "high_loss_samples.csv").exists()
    assert report.loaded_counts["false_negatives"] == 1
    assert report.loaded_counts["false_positives"] == 1
    assert report.used_for_oversampling_count == 2


def test_auto_falls_back_to_v1_hard_examples_when_no_v2_source_exists(tmp_path):
    hard_root = tmp_path / "artifacts" / "kaggle_v1_artifacts" / "outputs" / "hard_examples"
    _write_hard_example_csv(hard_root / "false_negatives.csv", ["v1_fn"])

    resolved = resolve_hard_example_sources(hard_example_source="auto", search_roots=[tmp_path])
    report = prepare_hard_example_report(
        hard_example_source=resolved.hard_examples_root,
        train_image_ids=["v1_fn"],
        validation_image_ids=[],
        strategy="analysis_only",
    )

    assert resolved.hard_examples_root == hard_root
    assert report.loaded_counts["false_negatives"] == 1
    assert report.used_for_oversampling_count == 0


def test_oversample_mode_changes_sampler_weights_with_fake_hard_examples(tmp_path):
    _write_hard_example_csv(tmp_path / "hard" / "false_negatives.csv", ["pos_hard"])
    examples = [_example("neg_a", 0), _example("neg_b", 0), _example("pos_hard", 1), _example("pos_easy", 1)]
    report = prepare_hard_example_report(
        hard_examples_root=tmp_path / "hard",
        train_image_ids=[example.image_id for example in examples],
        validation_image_ids=[],
        strategy="oversample",
    )

    weights = build_hard_example_weight_map(report, strategy="oversample")
    sampler = build_weighted_random_sampler(examples, seed=7, hard_example_weights=weights)

    assert report.used_for_oversampling_count > 0
    assert sampler.weights.tolist()[2] > sampler.weights.tolist()[3]


def test_oversampling_priority_order_is_fn_fp_uncertain_high_loss(tmp_path):
    _write_hard_example_csv(tmp_path / "hard" / "high_loss_samples.csv", ["high"])
    _write_hard_example_csv(tmp_path / "hard" / "uncertain.csv", ["uncertain"])
    _write_hard_example_csv(tmp_path / "hard" / "false_positives.csv", ["fp"])
    _write_hard_example_csv(tmp_path / "hard" / "false_negatives.csv", ["fn"])

    report = prepare_hard_example_report(
        hard_examples_root=tmp_path / "hard",
        train_image_ids=["fn", "fp", "uncertain", "high"],
        validation_image_ids=[],
        strategy="oversample",
    )

    assert report.oversampled_image_ids == ["fn", "fp", "uncertain", "high"]


def test_oversample_excludes_hard_examples_from_validation_before_reporting(tmp_path):
    _write_hard_example_csv(tmp_path / "hard" / "false_negatives.csv", ["val_hard"])
    split = SplitAssignment(
        train=[_example("train_a", 0), _example("train_b", 1)],
        validation=[_example("val_hard", 1), _example("val_easy", 0)],
    )
    initial = prepare_hard_example_report(
        hard_examples_root=tmp_path / "hard",
        train_image_ids=[example.image_id for example in split.train],
        validation_image_ids=[example.image_id for example in split.validation],
        strategy="analysis_only",
    )

    adjusted = exclude_hard_examples_from_validation(
        split,
        hard_example_image_ids=initial.validation_excluded_image_ids,
    )
    final = prepare_hard_example_report(
        hard_examples_root=tmp_path / "hard",
        train_image_ids=[example.image_id for example in adjusted.train],
        validation_image_ids=[example.image_id for example in adjusted.validation],
        strategy="oversample",
    )

    assert "val_hard" in [example.image_id for example in adjusted.train]
    assert "val_hard" not in [example.image_id for example in adjusted.validation]
    assert final.excluded_validation_count == 0
    assert final.used_for_oversampling_count == 1
    assert final.train_validation_disjoint is True


def test_hard_example_resolver_finds_v1_artifact_memory(tmp_path):
    hard_root = tmp_path / "artifacts" / "kaggle_v1_artifacts" / "outputs" / "hard_examples"
    _write_hard_example_csv(hard_root / "false_positives.csv", ["train_a"])
    _write_hard_example_csv(hard_root / "false_negatives.csv", ["train_b"])
    summary = hard_root.parent / "reports" / "hard_example_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        '{"counts":{"false_positives":1,"false_negatives":1,"uncertain":0,"high_loss_samples":0}}',
        encoding="utf-8",
    )

    resolved = resolve_hard_example_sources(search_roots=[tmp_path])
    report = prepare_hard_example_report(
        hard_examples_root=resolved.hard_examples_root,
        summary_path=resolved.summary_path,
        train_image_ids=["train_a", "train_b"],
        validation_image_ids=[],
        strategy="analysis_only",
    )

    assert resolved.hard_examples_root == hard_root
    assert report.loaded_counts["false_positives"] == 1
    assert report.loaded_counts["false_negatives"] == 1
    assert report.used_for_oversampling_count == 0


def test_v2b_he_output_paths_resolve_under_nested_kaggle_v2_root():
    output_root = Path("outputs/kaggle_v2/v2b_he_effnet_b1")

    assert _resolve_output_path(output_root, V2_MODEL_OUTPUT) == output_root / "kaggle_v2/models/classifier_best.pth"
    assert _resolve_output_path(output_root, V2_METRICS_OUTPUT) == output_root / "kaggle_v2/reports/classifier_metrics.json"
    assert _resolve_output_path(output_root, V2_THRESHOLD_OUTPUT) == output_root / "kaggle_v2/reports/best_threshold.json"
    assert _resolve_output_path(output_root, V2_PREDICTIONS_OUTPUT) == output_root / "kaggle_v2/predictions/val_classifier_predictions.csv"


def test_split_disjointness_is_reported():
    from src.training.train_classifier import SplitAssignment

    report = report_split_disjointness(
        SplitAssignment(
            train=[_example("a", 0), _example("b", 1)],
            validation=[_example("c", 0), _example("d", 1)],
        )
    )

    assert report["train_validation_disjoint"] is True
    assert report["overlap_image_ids"] == []


def test_v2_generated_artifacts_are_ignored_under_outputs():
    confidentiality = validate_generated_artifact_confidentiality([V2_MODEL_OUTPUT, V2_THRESHOLD_OUTPUT, V2_PREDICTIONS_OUTPUT])

    assert confidentiality.tracked_paths == []
    assert sorted(confidentiality.ignored_paths) == sorted([V2_MODEL_OUTPUT, V2_THRESHOLD_OUTPUT, V2_PREDICTIONS_OUTPUT])
