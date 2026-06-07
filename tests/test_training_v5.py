"""Tests for SPEC-010 V5 training contracts."""

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.models.classifier import create_classifier
from src.training.train_classifier import (
    TrainingRunConfig,
    TrainingValidationError,
    load_classifier_config,
    run_training,
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
            color = (25, 25, 25) if label == 1 else (220, 220, 220)
            Image.new("RGB", (48, 48), color).save(image_dir / image_id)

    with (dataset_root / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
    return dataset_root


def _make_hard_examples(root: Path, image_ids: list[str]) -> Path:
    hard_root = root / "hard_examples"
    hard_root.mkdir(parents=True)
    (hard_root / "false_negatives.csv").write_text(
        "image_id,true_label,probability,threshold,predicted_label\n"
        + "".join(f"{image_id},1,0.2,0.5,0\n" for image_id in image_ids),
        encoding="utf-8",
    )
    return hard_root


def test_v5_training_exports_required_validation_reports(tmp_path):
    dataset_root = _make_dataset(tmp_path)
    config = load_classifier_config("configs/v5_strong_classifier.yaml")
    config = TrainingRunConfig(**{**config.__dict__, "device": "cpu", "num_workers": 0, "epochs": 1, "batch_size": 2})

    result = run_training(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs" / "kaggle_v5" / "v5_strong_classifier",
        config=config,
        synthetic_smoke=True,
    )

    rows = list(csv.DictReader(result.predictions_path.open("r", newline="", encoding="utf-8")))
    threshold_report = json.loads(result.threshold_path.read_text(encoding="utf-8"))
    metrics_report = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert list(rows[0]) == ["image_id", "true_label", "prob_bad", "classifier_prediction", "target"]
    assert set(row["image_id"] for row in rows) == {example.image_id for example in result.split.validation}
    assert threshold_report["selection_source"] == "validation_only"
    assert threshold_report["split_source"] == "v2b_compatible"
    assert "confusion_counts" in threshold_report
    assert metrics_report["selected_model_name"] == "convnext_tiny"
    assert metrics_report["fallback_model_name"] == "efficientnet_b2"
    assert metrics_report["validation_target_distribution"] == metrics_report["validation_class_counts"]
    assert metrics_report["submission_row_count"] == 0
    assert metrics_report["public_score_baseline"] == pytest.approx(0.92181)
    assert metrics_report["v3_detector_public_score"] == pytest.approx(0.74169)
    assert metrics_report["selection_source"] == "validation_only"


def test_v5_training_falls_back_only_after_primary_failure(tmp_path, monkeypatch):
    dataset_root = _make_dataset(tmp_path)
    config = load_classifier_config("configs/v5_strong_classifier.yaml")
    config = TrainingRunConfig(**{**config.__dict__, "device": "cpu", "num_workers": 0, "epochs": 1, "batch_size": 2})
    calls: list[str] = []

    def _fake_create_classifier(*, model_name: str, num_classes: int, synthetic_smoke: bool):
        calls.append(model_name)
        if model_name == "convnext_tiny":
            raise RuntimeError("OOM")
        return create_classifier(model_name="tiny_cnn", num_classes=num_classes, synthetic_smoke=True)

    monkeypatch.setattr("src.training.train_classifier.create_classifier", _fake_create_classifier)

    result = run_training(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs" / "kaggle_v5" / "v5_strong_classifier",
        config=config,
        synthetic_smoke=False,
    )

    metrics_report = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert calls[:2] == ["convnext_tiny", "efficientnet_b2"]
    assert metrics_report["selected_model_name"] == "efficientnet_b2"


def test_v5_oversampling_requires_explicit_opt_in_and_reports_split_safety(tmp_path):
    dataset_root = _make_dataset(tmp_path, labels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    hard_root = _make_hard_examples(tmp_path, ["img_003.jpg", "img_009.jpg"])
    config = load_classifier_config("configs/v5_strong_classifier.yaml")
    config = TrainingRunConfig(
        **{
            **config.__dict__,
            "device": "cpu",
            "num_workers": 0,
            "epochs": 1,
            "batch_size": 2,
            "hard_example_strategy": "oversample",
            "hard_example_source": str(hard_root),
        }
    )

    result = run_training(
        dataset_root=dataset_root,
        output_root=tmp_path / "outputs" / "kaggle_v5" / "v5_strong_classifier",
        config=config,
        synthetic_smoke=True,
    )

    metrics_report = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    hard_report = metrics_report["split_disjointness"]["hard_examples"]
    assert metrics_report["hard_example_strategy"] == "oversample"
    assert hard_report["train_validation_disjoint"] is True
    assert "used_for_oversampling_count" in hard_report
    assert "validation_excluded_image_ids" in hard_report


def test_v5_rejects_public_score_as_training_signal():
    valid = load_classifier_config("configs/v5_strong_classifier.yaml")
    with pytest.raises(TrainingValidationError, match="0.92181"):
        config = TrainingRunConfig(**{**valid.__dict__, "public_score_baseline": 0.95})
        from src.training.train_classifier import _validate_training_config

        _validate_training_config(config)
