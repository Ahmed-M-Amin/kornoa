"""Synthetic tests for SPEC-005 classifier training."""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.training.metrics import compute_binary_metrics
from src.training.threshold_search import find_best_threshold
from src.training.train_classifier import (
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_PREDICTIONS_OUTPUT,
    DEFAULT_LABEL_DISTRIBUTION_OUTPUT,
    DEFAULT_SPLIT_DISTRIBUTION_OUTPUT,
    DEFAULT_THRESHOLD_OUTPUT,
    DEFAULT_METRICS_OUTPUT,
    TrainingRunConfig,
    TrainingValidationError,
    build_training_dataloaders,
    load_training_examples,
    make_stratified_split,
    run_training,
    select_training_device,
    validate_generated_artifact_confidentiality,
)


def _make_dataset(root: Path, *, labels: list[int] | None = None) -> Path:
    dataset_root = root / "classifier_cases"
    image_dir = dataset_root / "train_images"
    image_dir.mkdir(parents=True)
    (dataset_root / "test_images").mkdir(parents=True)
    labels = labels or [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    with (dataset_root / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        for index, label in enumerate(labels):
            image_id = f"img_{index:03d}.jpg"
            writer.writerow({"image_id": image_id, "target": label})
            color = (30, 30, 30) if label == 1 else (220, 220, 220)
            image = Image.new("RGB", (48, 48), color)
            image.save(image_dir / image_id)

    with (dataset_root / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "label"])
        writer.writeheader()

    return dataset_root


def _output_root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    (root / "models").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "predictions").mkdir(parents=True)
    return root


def test_classifier_config_uses_v1_defaults():
    with open("configs/classifier.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    classifier = config["classifier"]
    assert classifier["model_name"] == "efficientnet_b0"
    assert classifier["num_classes"] == 2
    assert classifier["image_size"] == 384
    assert classifier["validation_split"] == 0.2
    assert classifier["seed"] == 42


def test_load_training_examples_rejects_invalid_binary_labels(tmp_path):
    dataset_root = _make_dataset(tmp_path, labels=[0, 1, 2, 1, 0, 1])

    with pytest.raises(TrainingValidationError, match="binary values 0 or 1"):
        load_training_examples(dataset_root)


def test_load_training_examples_reports_missing_images(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    (dataset_root / "train_images" / "img_000.jpg").unlink()

    with pytest.raises(TrainingValidationError, match="Missing 1 training images: img_000.jpg"):
        load_training_examples(dataset_root)


def test_missing_image_diagnostics_are_limited_to_first_twenty(tmp_path):
    labels = [0, 1] * 15
    dataset_root = _make_dataset(tmp_path, labels=labels)
    for image_path in sorted((dataset_root / "train_images").glob("*.jpg")):
        image_path.unlink()

    with pytest.raises(TrainingValidationError) as exc_info:
        load_training_examples(dataset_root)

    message = str(exc_info.value)
    assert message.startswith("Missing 30 training images: ")
    assert "img_000.jpg" in message
    assert "img_019.jpg" in message
    assert "img_020.jpg" not in message
    assert "and 10 more" in message


def test_one_class_training_data_is_rejected(tmp_path):
    dataset_root = _make_dataset(tmp_path, labels=[0, 0, 0, 0, 0])
    examples = load_training_examples(dataset_root)

    with pytest.raises(TrainingValidationError, match="both binary classes"):
        make_stratified_split(examples, validation_split=0.2, seed=42)


def test_stratified_split_is_reproducible_and_uses_80_20(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    examples = load_training_examples(dataset_root)

    first = make_stratified_split(examples, validation_split=0.2, seed=42)
    second = make_stratified_split(examples, validation_split=0.2, seed=42)

    assert [item.image_id for item in first.train] == [item.image_id for item in second.train]
    assert [item.image_id for item in first.validation] == [item.image_id for item in second.validation]
    assert len(first.train) == 8
    assert len(first.validation) == 2
    assert {item.label for item in first.validation} == {0, 1}


def test_binary_metrics_include_f1_confusion_and_class_counts():
    metrics = compute_binary_metrics(
        y_true=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.6, 0.4],
        threshold=0.5,
    )

    assert metrics.f1_score == pytest.approx(0.5)
    assert metrics.confusion_counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert metrics.class_counts == {"0": 2, "1": 2}


def test_threshold_search_uses_lowest_threshold_when_f1_ties():
    result = find_best_threshold(
        y_true=[0, 1],
        probabilities=[0.2, 0.8],
        candidates=[0.3, 0.5, 0.7],
    )

    assert result.threshold == pytest.approx(0.3)
    assert result.f1_score == pytest.approx(1.0)
    assert result.tie_break == "lowest_threshold"


def test_training_uses_spec004_normalized_preprocessing(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    examples = load_training_examples(dataset_root)

    sample = examples[0].load_preprocessed(split="validation")

    assert sample.shape == (3, 384, 384)
    assert sample.dtype == np.float32


def test_training_dataloaders_use_configured_batch_size_and_split_order(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    split = make_stratified_split(load_training_examples(dataset_root), validation_split=0.2, seed=42)
    config = TrainingRunConfig(batch_size=3, num_workers=0)

    loaders = build_training_dataloaders(split, config=config, device=torch.device("cpu"), synthetic_smoke=True)

    train_inputs, train_labels, train_image_ids = next(iter(loaders.train))
    validation_inputs, validation_labels, validation_image_ids = next(iter(loaders.validation))
    assert train_inputs.shape == (3, 3, 384, 384)
    assert train_labels.shape == (3,)
    assert len(train_image_ids) == 3
    assert validation_inputs.shape[1:] == (3, 384, 384)
    assert validation_labels.shape == (len(split.validation),)
    assert len(validation_image_ids) == len(split.validation)
    assert loaders.train.batch_size == 3
    assert loaders.validation.batch_size == 3
    assert type(loaders.train.sampler).__name__ == "RandomSampler"
    assert type(loaders.validation.sampler).__name__ == "SequentialSampler"


def test_debug_sample_limits_preserve_stratified_split(tmp_path):
    dataset_root = _make_dataset(tmp_path, labels=[0] * 8 + [1] * 8)
    examples = load_training_examples(dataset_root)

    split = make_stratified_split(
        examples,
        validation_split=0.25,
        seed=42,
        max_train_samples=4,
        max_val_samples=2,
    )

    assert len(split.train) == 4
    assert len(split.validation) == 2
    assert {item.label for item in split.train} == {0, 1}
    assert {item.label for item in split.validation} == {0, 1}


def test_training_device_selection_supports_explicit_cpu():
    device = select_training_device("cpu")

    assert device.type == "cpu"


def test_synthetic_training_metrics_record_device(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    result = run_training(
        dataset_root=dataset_root,
        output_root=output_root,
        config=TrainingRunConfig(device="cpu", num_workers=0),
        synthetic_smoke=True,
        seed=42,
        epochs=1,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["device"] == "cpu"


def test_distribution_reports_are_saved(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    result = run_training(
        dataset_root=dataset_root,
        output_root=output_root,
        config=TrainingRunConfig(device="cpu", num_workers=0),
        synthetic_smoke=True,
        seed=42,
        epochs=1,
    )

    label_distribution = json.loads(result.label_distribution_path.read_text(encoding="utf-8"))
    split_distribution = json.loads(result.split_distribution_path.read_text(encoding="utf-8"))
    assert label_distribution["total_rows"] == 10
    assert label_distribution["label_counts"] == {"0": 5, "1": 5}
    assert split_distribution["train"]["count"] == len(result.split.train)
    assert split_distribution["validation"]["count"] == len(result.split.validation)
    assert split_distribution["train"]["label_counts"] == {"0": 4, "1": 4}
    assert split_distribution["validation"]["label_counts"] == {"0": 1, "1": 1}


def test_progress_logging_does_not_break_training(tmp_path, capsys):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    run_training(
        dataset_root=dataset_root,
        output_root=output_root,
        config=TrainingRunConfig(device="cpu", num_workers=0, log_every_n_batches=1),
        synthetic_smoke=True,
        seed=42,
        epochs=1,
    )

    captured = capsys.readouterr()
    assert "Training start" in captured.out
    assert "DataLoader created" in captured.out
    assert "epoch=1 batch=1" in captured.out
    assert "validation_f1=" in captured.out
    assert "total_runtime_seconds=" in captured.out


def test_class_weights_are_calculated_from_train_split_only(tmp_path):
    dataset_root = _make_dataset(tmp_path, labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    output_root = _output_root(tmp_path)

    result = run_training(
        dataset_root=dataset_root,
        output_root=output_root,
        config=TrainingRunConfig(device="cpu", num_workers=0),
        synthetic_smoke=True,
        seed=42,
        epochs=1,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    train_counts = metrics["train_class_counts"]
    assert metrics["pos_weight"] == pytest.approx(train_counts["0"] / train_counts["1"])


def test_threshold_search_uses_validation_probabilities_only(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    result = run_training(
        dataset_root=dataset_root,
        output_root=output_root,
        config=TrainingRunConfig(device="cpu", num_workers=0),
        synthetic_smoke=True,
        seed=42,
        epochs=1,
    )

    prediction_rows = list(csv.DictReader(result.predictions_path.open("r", newline="", encoding="utf-8")))
    threshold_record = json.loads(result.threshold_path.read_text(encoding="utf-8"))
    recomputed = find_best_threshold(
        y_true=[int(row["true_label"]) for row in prediction_rows],
        probabilities=[float(row["probability"]) for row in prediction_rows],
    )
    assert threshold_record["threshold"] == pytest.approx(recomputed.threshold)


def test_synthetic_training_smoke_saves_minimal_artifacts_under_one_minute(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    started = time.perf_counter()
    result = run_training(
        dataset_root=dataset_root,
        output_root=output_root,
        synthetic_smoke=True,
        seed=42,
        epochs=2,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 60
    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert result.threshold_path.exists()
    assert result.predictions_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "f1_score" in metrics
    assert "train_class_counts" in metrics
    assert "validation_class_counts" in metrics

    with result.predictions_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.split.validation)
    assert {"image_id", "true_label", "probability", "threshold", "predicted_label"}.issubset(rows[0])


def test_generated_training_artifacts_are_ignored_when_under_outputs():
    paths = [
        DEFAULT_MODEL_OUTPUT,
        DEFAULT_METRICS_OUTPUT,
        DEFAULT_THRESHOLD_OUTPUT,
        DEFAULT_PREDICTIONS_OUTPUT,
        DEFAULT_LABEL_DISTRIBUTION_OUTPUT,
        DEFAULT_SPLIT_DISTRIBUTION_OUTPUT,
    ]

    confidentiality = validate_generated_artifact_confidentiality(paths)

    assert confidentiality.tracked_paths == []
    assert sorted(confidentiality.ignored_paths) == sorted(paths)


def test_training_cli_synthetic_smoke(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.training.train_classifier",
            "--config",
            "configs/classifier.yaml",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--synthetic-smoke",
            "--epochs",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_root / "models" / "classifier_effnet_b0_best.pth").exists()


def test_no_out_of_scope_outputs_are_created_by_training(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    output_root = _output_root(tmp_path)

    run_training(dataset_root=dataset_root, output_root=output_root, synthetic_smoke=True, seed=42, epochs=1)

    forbidden = [
        output_root / "submissions" / "submission.csv",
        output_root / "figures" / "gradcam.png",
        output_root / "predictions" / "test_predictions.csv",
    ]
    assert not any(path.exists() for path in forbidden)
