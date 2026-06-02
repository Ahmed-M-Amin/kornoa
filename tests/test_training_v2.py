"""Tests for SPEC-007 V2 training contracts."""

from pathlib import Path

import pytest

from src.training.hard_example_mining import prepare_hard_example_report
from src.training.train_classifier import (
    V2_MODEL_OUTPUT,
    V2_PREDICTIONS_OUTPUT,
    V2_THRESHOLD_OUTPUT,
    TrainingExample,
    TrainingRunConfig,
    TrainingValidationError,
    build_hard_example_weight_map,
    build_weighted_random_sampler,
    load_classifier_config,
    report_split_disjointness,
    validate_generated_artifact_confidentiality,
    validate_v2_scope_guards,
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
