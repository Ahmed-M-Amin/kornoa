"""Tests for the V2B-enhanced TTA ladder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_tta_config(tmp_path: Path) -> Path:
    val_predictions = _write_csv(
        tmp_path / "inputs" / "val_predictions.csv",
        [
            {"image_id": "a.jpg", "true_label": 0, "prob_bad": 0.18, "prob_bad_hflip": 0.20},
            {"image_id": "b.jpg", "true_label": 1, "prob_bad": 0.47, "prob_bad_hflip": 0.62},
            {"image_id": "c.jpg", "true_label": 1, "prob_bad": 0.80, "prob_bad_hflip": 0.78},
            {"image_id": "d.jpg", "true_label": 0, "prob_bad": 0.55, "prob_bad_hflip": 0.40},
        ],
    )
    test_predictions = _write_csv(
        tmp_path / "inputs" / "test_predictions.csv",
        [
            {"image_id": "ta.jpg", "prob_bad": 0.20, "prob_bad_hflip": 0.22},
            {"image_id": "tb.jpg", "prob_bad": 0.58, "prob_bad_hflip": 0.66},
        ],
    )
    config = {
        "paths": {
            "dataset_root": "1st-krones-vision-ai-challenge",
            "model_checkpoint": "outputs/kaggle_v2b_enhanced/448/models/classifier_best.pth",
            "model_config": "configs/v2b_enhanced_448.yaml",
            "output_root": str(tmp_path / "outputs" / "kaggle_v2b_enhanced" / "tta"),
            "validation_predictions": str(val_predictions),
            "test_predictions": str(test_predictions),
        },
        "model": {"model_name": "efficientnet_b1", "image_size": 448},
        "tta": {"modes": ["no_tta", "hflip_tta", "gated_hflip_tta"], "margins": [0.04, 0.06, 0.08]},
        "threshold_search": {
            "start": 0.15,
            "end": 0.60,
            "step": 0.01,
            "positive_class": 1,
            "zero_division": 0,
        },
        "output": {
            "save_validation_predictions": True,
            "save_selected_submission": True,
            "save_all_candidates_report": True,
        },
    }
    path = tmp_path / "configs" / "v2b_enhanced_tta.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_config_loads_required_values():
    from src.inference.v2b_enhanced_tta import load_tta_config

    config = load_tta_config("configs/v2b_enhanced_tta.yaml")

    assert config.model_name == "efficientnet_b1"
    assert config.image_size == 448
    assert config.modes == ("no_tta", "hflip_tta", "gated_hflip_tta")
    assert config.margins == (0.04, 0.06, 0.08)
    assert config.threshold_start == pytest.approx(0.15)
    assert config.threshold_end == pytest.approx(0.60)
    assert config.positive_class == 1
    assert config.zero_division == 0


def test_v2b_enhanced_training_configs_load_with_required_ladder_values():
    from src.training.train_classifier import load_classifier_config

    cfg_384 = load_classifier_config("configs/v2b_enhanced_384.yaml")
    cfg_448 = load_classifier_config("configs/v2b_enhanced_448.yaml")

    assert cfg_384.v2 is True
    assert cfg_384.model_name == "efficientnet_b1"
    assert cfg_384.start_checkpoint.endswith("classifier_best.pth")
    assert cfg_384.image_size == 384
    assert cfg_384.epochs == 4
    assert cfg_384.learning_rate == pytest.approx(0.00003)
    assert cfg_384.imbalance_strategy == "bce"
    assert cfg_384.weighted_sampler is False
    assert cfg_384.augmentation_recipe == "mild_safe"
    assert cfg_384.split_source == "v2b_compatible"

    assert cfg_448.v2 is True
    assert cfg_448.model_name == "efficientnet_b1"
    assert cfg_448.start_checkpoint == "outputs/kaggle_v2b_enhanced/384/models/classifier_best.pth"
    assert cfg_448.image_size == 448
    assert cfg_448.epochs == 2
    assert cfg_448.learning_rate == pytest.approx(0.00001)
    assert cfg_448.imbalance_strategy == "bce"
    assert cfg_448.weighted_sampler is False
    assert cfg_448.augmentation_recipe == "mild_safe"
    assert cfg_448.split_source == "v2b_compatible"


def test_start_checkpoint_loader_applies_tiny_classifier_weights(tmp_path):
    from src.models.classifier import create_classifier
    from src.training.train_classifier import TrainingRunConfig, load_start_checkpoint_if_configured

    checkpoint_model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    for parameter in checkpoint_model.parameters():
        parameter.data.fill_(0.25)
    checkpoint_path = tmp_path / "classifier_best.pth"
    torch.save({"model_state_dict": checkpoint_model.state_dict()}, checkpoint_path)

    model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    for parameter in model.parameters():
        parameter.data.zero_()

    config = TrainingRunConfig(
        model_name="efficientnet_b1",
        experiment_name="v2b_enhanced_384",
        v2=True,
        start_checkpoint=str(checkpoint_path),
        imbalance_strategy="bce",
        weighted_sampler=False,
        hard_example_strategy="none",
        augmentation_recipe="mild_safe",
    )
    loaded = load_start_checkpoint_if_configured(model, config, torch.device("cpu"))

    assert loaded == checkpoint_path
    assert all(torch.allclose(parameter, torch.full_like(parameter, 0.25)) for parameter in model.parameters())


def test_missing_configured_start_checkpoint_fails_clearly(tmp_path):
    from src.models.classifier import create_classifier
    from src.training.train_classifier import TrainingRunConfig, load_start_checkpoint_if_configured

    model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    missing = tmp_path / "missing.pth"
    config = TrainingRunConfig(
        model_name="efficientnet_b1",
        experiment_name="v2b_enhanced_384",
        v2=True,
        start_checkpoint=str(missing),
        imbalance_strategy="bce",
        weighted_sampler=False,
        hard_example_strategy="none",
        augmentation_recipe="mild_safe",
    )

    with pytest.raises(FileNotFoundError, match="Configured start_checkpoint not found"):
        load_start_checkpoint_if_configured(model, config, torch.device("cpu"))


def test_v2b_enhanced_training_requires_loaded_start_checkpoint():
    from src.models.classifier import create_classifier
    from src.training.train_classifier import TrainingRunConfig, TrainingValidationError, load_start_checkpoint_if_configured

    model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    config = TrainingRunConfig(
        model_name="efficientnet_b1",
        experiment_name="v2b_enhanced_384",
        v2=True,
        start_checkpoint="",
        imbalance_strategy="bce",
        weighted_sampler=False,
        hard_example_strategy="none",
        augmentation_recipe="mild_safe",
    )

    with pytest.raises(TrainingValidationError, match="requires start_checkpoint"):
        load_start_checkpoint_if_configured(model, config, torch.device("cpu"))


def test_supported_modes_margins_and_threshold_candidates():
    from src.inference.v2b_enhanced_tta import SUPPORTED_TTA_MODES, threshold_candidates

    assert {"no_tta", "hflip_tta", "gated_hflip_tta"} <= SUPPORTED_TTA_MODES
    assert threshold_candidates(0.15, 0.60, 0.01)[0] == pytest.approx(0.15)
    assert threshold_candidates(0.15, 0.60, 0.01)[-1] == pytest.approx(0.60)


def test_tta_search_writes_required_reports_predictions_and_selected_submission(tmp_path):
    from src.inference.v2b_enhanced_tta import REQUIRED_CANDIDATE_FIELDS, run_tta_search

    config_path = _write_tta_config(tmp_path)
    result = run_tta_search(config_path)

    output_root = tmp_path / "outputs" / "kaggle_v2b_enhanced" / "tta"
    grid = pd.read_csv(output_root / "reports" / "tta_search_grid.csv")
    summary = json.loads((output_root / "reports" / "tta_candidate_summary.json").read_text(encoding="utf-8"))
    val_rows = list(csv.DictReader((output_root / "predictions" / "val_predictions_selected.csv").open(newline="", encoding="utf-8")))
    submission_rows = list(
        csv.DictReader((output_root / "submissions" / "submission_v2b_enhanced_tta.csv").open(newline="", encoding="utf-8"))
    )

    assert set(REQUIRED_CANDIDATE_FIELDS) <= set(grid.columns)
    assert summary["uses_test_labels"] is False
    assert summary["selected_candidate"]["mode"] == result.selected_candidate.mode
    assert result.selected_submission_path == output_root / "submissions" / "submission_v2b_enhanced_tta.csv"
    assert list(submission_rows[0]) == ["image_id", "target"]
    assert "true_label" not in submission_rows[0]
    assert {"image_id", "true_label", "prob_bad", "target", "mode", "margin", "threshold"} <= set(val_rows[0])


def test_candidate_metrics_include_runtime_tta_usage_and_changed_counts(tmp_path):
    from src.inference.v2b_enhanced_tta import run_tta_search

    config_path = _write_tta_config(tmp_path)
    result = run_tta_search(config_path)

    selected = result.selected_candidate
    assert selected.runtime_seconds >= 0.0
    assert selected.images_per_second > 0.0
    assert selected.tta_usage_count >= 0
    assert 0.0 <= selected.tta_usage_ratio <= 1.0
    assert selected.changed_count_vs_no_tta >= 0
    assert 0.0 <= selected.changed_ratio_vs_no_tta <= 1.0


def test_test_prediction_input_rejects_label_columns(tmp_path):
    from src.inference.v2b_enhanced_tta import V2BEnhancedTtaError, run_tta_search

    config_path = _write_tta_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    labeled_test = _write_csv(
        tmp_path / "inputs" / "bad_test_predictions.csv",
        [{"image_id": "ta.jpg", "true_label": 1, "prob_bad": 0.9, "prob_bad_hflip": 0.8}],
    )
    config["paths"]["test_predictions"] = str(labeled_test)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(V2BEnhancedTtaError, match="test labels"):
        run_tta_search(config_path)
