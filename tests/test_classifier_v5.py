"""Tests for SPEC-010 V5 classifier config and transform contracts."""

from pathlib import Path

import pytest

from src.data.transforms import create_preprocessing_profile
from src.models.classifier import validate_v5_classifier_scope
from src.training.train_classifier import (
    V5_METRICS_OUTPUT,
    V5_MODEL_OUTPUT,
    V5_PREDICTIONS_OUTPUT,
    V5_THRESHOLD_OUTPUT,
    TrainingRunConfig,
    TrainingValidationError,
    _resolve_output_path,
    _validate_training_config,
    load_classifier_config,
    validate_generated_artifact_confidentiality,
)


def test_v5_config_defaults_and_output_roots():
    config = load_classifier_config("configs/v5_strong_classifier.yaml")

    assert config.v5 is True
    assert config.model_name == "convnext_tiny"
    assert config.fallback_model_name == "efficientnet_b2"
    assert config.image_size == 512
    assert config.split_source == "v2b_compatible"
    assert config.hard_example_strategy == "analysis_only"
    assert config.augmentation_recipe == "v5_safe"
    assert _resolve_output_path(Path("outputs/kaggle_v5/v5_strong_classifier"), V5_MODEL_OUTPUT) == Path(
        "outputs/kaggle_v5/v5_strong_classifier/models/classifier_best.pth"
    )
    assert _resolve_output_path(Path("outputs/kaggle_v5/v5_strong_classifier"), V5_METRICS_OUTPUT) == Path(
        "outputs/kaggle_v5/v5_strong_classifier/reports/classifier_metrics.json"
    )
    assert _resolve_output_path(Path("outputs/kaggle_v5/v5_strong_classifier"), V5_THRESHOLD_OUTPUT) == Path(
        "outputs/kaggle_v5/v5_strong_classifier/reports/best_threshold.json"
    )
    assert _resolve_output_path(Path("outputs/kaggle_v5/v5_strong_classifier"), V5_PREDICTIONS_OUTPUT) == Path(
        "outputs/kaggle_v5/v5_strong_classifier/predictions/val_classifier_predictions.csv"
    )


def test_v5_model_scope_is_bounded():
    validate_v5_classifier_scope("convnext_tiny", "efficientnet_b2")

    with pytest.raises(ValueError, match="convnext_tiny"):
        validate_v5_classifier_scope("efficientnet_b1", "efficientnet_b2")
    with pytest.raises(ValueError, match="efficientnet_b2"):
        validate_v5_classifier_scope("convnext_tiny", "convnext_tiny")


def test_v5_training_validation_rejects_invalid_scope():
    with pytest.raises(TrainingValidationError, match="convnext_tiny"):
        _validate_training_config(TrainingRunConfig(v5=True, model_name="efficientnet_b1"))
    valid = load_classifier_config("configs/v5_strong_classifier.yaml")
    with pytest.raises(TrainingValidationError, match="efficientnet_b2"):
        _validate_training_config(TrainingRunConfig(**{**valid.__dict__, "fallback_model_name": "efficientnet_b1"}))
    with pytest.raises(TrainingValidationError, match="split_source"):
        _validate_training_config(TrainingRunConfig(**{**valid.__dict__, "split_source": "missing_split_reference"}))


def test_v5_safe_augmentation_is_stronger_but_bounded():
    v2_profile = create_preprocessing_profile("train", target_size=(512, 512), seed=42, augmentation_recipe="v2_safe")
    v5_profile = create_preprocessing_profile("train", target_size=(512, 512), seed=42, augmentation_recipe="v5_safe")

    assert v5_profile.target_size == (512, 512)
    assert v5_profile.max_rotation_degrees >= v2_profile.max_rotation_degrees
    assert v5_profile.max_shift_ratio >= v2_profile.max_shift_ratio
    assert v5_profile.brightness_delta <= v2_profile.brightness_delta
    assert v5_profile.contrast_delta <= v2_profile.contrast_delta
    assert v5_profile.blur_probability <= v2_profile.blur_probability
    assert v5_profile.noise_std <= v2_profile.noise_std


def test_v5_generated_artifacts_are_ignored():
    confidentiality = validate_generated_artifact_confidentiality(
        [V5_MODEL_OUTPUT, V5_METRICS_OUTPUT, V5_THRESHOLD_OUTPUT, V5_PREDICTIONS_OUTPUT]
    )

    assert confidentiality.tracked_paths == []
    assert sorted(confidentiality.ignored_paths) == sorted(
        [V5_MODEL_OUTPUT, V5_METRICS_OUTPUT, V5_THRESHOLD_OUTPUT, V5_PREDICTIONS_OUTPUT]
    )
