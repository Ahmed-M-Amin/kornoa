"""Tests for V2.2 hard-example training support."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml


def _write_rows(path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = columns or ["image_id", "error_type", "true_label", "v2b_prediction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_sources(root: Path) -> dict[str, Path]:
    hard_negatives = root / "hard_negatives.csv"
    hard_positives = root / "hard_positives.csv"
    uncertain = root / "uncertain.csv"
    _write_rows(
        hard_negatives,
        [
            {"image_id": "fp_a.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1},
            {"image_id": "fp_a.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1},
            {"image_id": 123, "error_type": "FP", "true_label": 0, "v2b_prediction": 1},
        ],
    )
    _write_rows(
        hard_positives,
        [{"image_id": "fn_a.png", "error_type": "FN", "true_label": 1, "v2b_prediction": 0}],
    )
    _write_rows(
        uncertain,
        [{"image_id": "uncertain_a.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}],
    )
    return {
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "uncertain_examples": uncertain,
    }


def test_hard_example_loader_loads_counts_and_deduplicates_image_ids(tmp_path):
    from src.data.hard_examples import load_hard_examples

    paths = _write_sources(tmp_path)

    report = load_hard_examples(**paths)

    assert report.hard_negative_count == 2
    assert report.hard_positive_count == 1
    assert report.uncertain_count == 1
    assert report.hard_negative_image_ids == ["fp_a.png", "123"]
    assert report.hard_positive_image_ids == ["fn_a.png"]
    assert report.strong_hard_example_image_ids == ["fp_a.png", "123", "fn_a.png"]
    assert "uncertain_a.png" not in report.strong_hard_example_image_ids


def test_hard_example_loader_missing_csv_fails_clearly(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    paths["hard_negatives"].unlink()

    with pytest.raises(HardExampleError, match="Hard-example CSV not found"):
        load_hard_examples(**paths)


def test_hard_example_loader_missing_image_id_column_fails_clearly(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    _write_rows(paths["hard_negatives"], [{"error_type": "FP", "true_label": 0, "v2b_prediction": 1}], columns=["error_type", "true_label", "v2b_prediction"])

    with pytest.raises(HardExampleError, match="missing required columns: image_id"):
        load_hard_examples(**paths)


def test_hard_example_loader_rejects_wrong_hard_negative_logic(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    _write_rows(paths["hard_negatives"], [{"image_id": "bad.png", "error_type": "FN", "true_label": 1, "v2b_prediction": 0}])

    with pytest.raises(HardExampleError, match="hard negatives must be FP rows"):
        load_hard_examples(**paths)


def test_hard_example_loader_rejects_wrong_hard_positive_logic(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    _write_rows(paths["hard_positives"], [{"image_id": "bad.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}])

    with pytest.raises(HardExampleError, match="hard positives must be FN rows"):
        load_hard_examples(**paths)


def test_v2_2_config_is_safe_and_uses_binary_f1_contract():
    config_path = Path("configs/v2_2_hard_examples.yaml")

    assert config_path.exists()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hard_examples = payload["hard_examples"]
    metrics = payload["metrics"]

    assert payload["output"]["root"] == "outputs/kaggle_v2_2/v2_2_hard_examples"
    assert hard_examples["enabled"] is True
    assert hard_examples["hard_negatives"]
    assert hard_examples["hard_positives"]
    assert hard_examples["uncertain_examples"]
    assert hard_examples["strategy"] == "conservative_loss_weighting"
    assert 1.25 <= hard_examples["hard_negative_weight"] <= 1.5
    assert 1.25 <= hard_examples["hard_positive_weight"] <= 1.5
    assert hard_examples["max_extra_sampling_multiplier"] <= 2.0
    assert payload["safety"]["generate_submission"] is False
    assert payload["safety"]["allow_test_labels"] is False
    assert metrics["positive_class"] == 1
    assert metrics["zero_division"] == 0
    assert payload["training"]["hard_example_strategy"] == "conservative_loss_weighting"


def test_training_cli_accepts_v2_2_smoke_aliases_without_starting_real_training(monkeypatch):
    from src.training import train_classifier

    captured = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)

        class _Result:
            class metrics:
                f1_score = 0.0

            class threshold:
                threshold = 0.5

            model_path = Path("unused.pth")

        return _Result()

    monkeypatch.setattr(train_classifier, "run_training", _fake_run_training)

    exit_code = train_classifier.main(
        [
            "--config",
            "configs/v2_2_hard_examples.yaml",
            "--max-epochs",
            "1",
            "--limit-train-batches",
            "10",
            "--limit-val-batches",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured["epochs"] == 1
    assert captured["config"].limit_train_batches == 10
    assert captured["config"].limit_val_batches == 5
    assert captured["config"].hard_examples_enabled is True
    assert captured["config"].hard_example_strategy == "conservative_loss_weighting"
    assert captured["dataset_root"] == "1st-krones-vision-ai-challenge"
    assert captured["output_root"] == "outputs/kaggle_v2_2/v2_2_hard_examples"


def test_v2_2_output_paths_are_isolated_from_old_v2_outputs():
    from src.training.train_classifier import (
        V2_2_MODEL_OUTPUT,
        V2_2_METRICS_OUTPUT,
        V2_2_PREDICTIONS_OUTPUT,
        V2_2_THRESHOLD_OUTPUT,
        _default_model_output_for_config,
        _default_predictions_output_for_config,
        _default_threshold_output_for_config,
        _resolve_output_path,
        load_classifier_config,
    )

    config = load_classifier_config("configs/v2_2_hard_examples.yaml")
    output_root = Path(config.output_root)

    assert _default_model_output_for_config(config) == V2_2_MODEL_OUTPUT
    assert _default_threshold_output_for_config(config) == V2_2_THRESHOLD_OUTPUT
    assert _default_predictions_output_for_config(config) == V2_2_PREDICTIONS_OUTPUT
    assert _resolve_output_path(output_root, V2_2_MODEL_OUTPUT) == output_root / "models/classifier_best.pth"
    assert _resolve_output_path(output_root, V2_2_METRICS_OUTPUT) == output_root / "reports/classifier_metrics.json"
    assert _resolve_output_path(output_root, V2_2_THRESHOLD_OUTPUT) == output_root / "reports/best_threshold.json"
    assert _resolve_output_path(output_root, V2_2_PREDICTIONS_OUTPUT) == output_root / "predictions/val_classifier_predictions.csv"
    assert "kaggle_v2_2" in str(_default_model_output_for_config(config))
    assert "kaggle_v2/models" not in str(_default_model_output_for_config(config)).replace("\\", "/")


def test_conservative_weight_map_weights_only_strong_hard_examples(tmp_path):
    from src.data.hard_examples import load_hard_examples
    from src.training.train_classifier import (
        TrainingRunConfig,
        build_conservative_hard_example_weight_map,
    )

    paths = _write_sources(tmp_path)
    report = load_hard_examples(**paths)
    config = TrainingRunConfig(
        hard_example_strategy="conservative_loss_weighting",
        hard_negative_weight=1.35,
        hard_positive_weight=1.4,
        uncertain_weight=1.0,
    )

    weights = build_conservative_hard_example_weight_map(
        report,
        train_image_ids=["fp_a.png", "fn_a.png", "uncertain_a.png", "easy.png"],
        config=config,
    )

    assert weights == {"fp_a.png": 1.35, "fn_a.png": 1.4}
    assert weights.get("uncertain_a.png", 1.0) == 1.0
    assert weights.get("easy.png", 1.0) == 1.0


def test_conservative_weighted_dataset_returns_per_sample_weights():
    from src.training.train_classifier import (
        ClassifierTrainingDataset,
    )

    class _Example:
        def __init__(self, image_id: str, label: int) -> None:
            self.image_id = image_id
            self.label = label

        def load_preprocessed(self, **_: object) -> np.ndarray:
            return np.zeros((3, 384, 384), dtype=np.float32)

    dataset = ClassifierTrainingDataset(
        [
            _Example("hard.png", 1),
            _Example("easy.png", 0),
        ],
        split_name="train",
        sample_weights={"hard.png": 1.35},
    )

    hard = dataset[0]
    easy = dataset[1]

    assert len(hard) == 4
    assert hard[3].item() == pytest.approx(1.35)
    assert easy[3].item() == pytest.approx(1.0)


def test_v2_2_config_rejects_aggressive_hard_example_weights():
    from src.training.train_classifier import TrainingRunConfig, TrainingValidationError, _validate_training_config

    config = TrainingRunConfig(
        v2=True,
        experiment_name="v2_2_hard_examples",
        hard_examples_enabled=True,
        hard_example_strategy="conservative_loss_weighting",
        hard_negatives_path="hn.csv",
        hard_positives_path="hp.csv",
        hard_negative_weight=2.1,
        hard_positive_weight=1.35,
    )

    with pytest.raises(TrainingValidationError, match="weights"):
        _validate_training_config(config)
