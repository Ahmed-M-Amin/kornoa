"""Tests for the V1R repaired baseline configuration."""

from __future__ import annotations

import yaml

from src.training.train_classifier import load_classifier_config


def _load_v1r_yaml() -> dict:
    with open("configs/v1r_repaired_baseline.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_v1r_config_file_exists_and_defines_required_model_settings():
    config = _load_v1r_yaml()

    assert config["model"]["model_name"] == "efficientnet_b0"
    assert config["model"]["image_size"] in {320, 384}
    assert config["model"]["image_size"] == 384
    assert config["model"]["pretrained"] is True


def test_v1r_training_defaults_are_cheap_and_validation_only():
    config = _load_v1r_yaml()
    training = config["training"]

    assert training["epochs"] == 4
    assert training["learning_rate"] == 0.0001
    assert training["loss"] == "weighted_bce"
    assert training["weighted_sampler"] is False
    assert training["augmentation_recipe"] == "mild_safe"
    assert training["split_source"] == "v2b_compatible"
    assert training["save_validation_predictions"] is True
    assert training["save_test_predictions"] is False
    assert training["threshold_selection"] == "validation_f1_binary_pos_label_1"


def test_v1r_output_root_is_runtime_only_location():
    config = _load_v1r_yaml()

    assert config["output"]["output_root"] == "outputs/kaggle_v1r/v1r_repaired_baseline"


def test_v1r_config_loads_through_classifier_training_config():
    config = load_classifier_config("configs/v1r_repaired_baseline.yaml")

    assert config.model_name == "efficientnet_b0"
    assert config.image_size == 384
    assert config.epochs == 4
    assert config.learning_rate == 0.0001
    assert config.imbalance_strategy == "weighted_bce"
    assert config.weighted_sampler is False
    assert config.augmentation_recipe == "mild_safe"
    assert config.split_source == "v2b_compatible"


def test_v1r_mild_safe_augmentation_recipe_is_supported():
    from src.data.transforms import create_preprocessing_profile

    profile = create_preprocessing_profile("train", augmentation_recipe="mild_safe", seed=42)

    assert profile.augment is True
    assert profile.max_rotation_degrees <= 5
