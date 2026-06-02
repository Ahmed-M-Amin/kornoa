"""Tests for SPEC-006 V1 prediction helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.models.classifier import create_classifier


def _make_tiny_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), path)
    return path


def _make_artifact_root(root: Path) -> Path:
    artifact_root = root / "artifacts" / "kaggle_v1_artifacts" / "outputs" / "kaggle_v1"
    _make_tiny_checkpoint(artifact_root / "models" / "classifier_effnet_b0_best.pth")
    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "best_threshold.json").write_text(json.dumps({"threshold": 0.6}), encoding="utf-8")
    (reports / "classifier_metrics.json").write_text(json.dumps({"f1_score": 0.9}), encoding="utf-8")
    predictions = artifact_root / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    (predictions / "val_classifier_predictions.csv").write_text(
        "image_id,true_label,probability,threshold,predicted_label\n",
        encoding="utf-8",
    )
    return artifact_root


def _make_image(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)
    return path


def test_resolve_artifact_root_accepts_concrete_and_parent_roots(tmp_path):
    from src.inference.predict import resolve_v1_artifact_paths

    artifact_root = _make_artifact_root(tmp_path)
    concrete = resolve_v1_artifact_paths(artifact_root)
    parent = resolve_v1_artifact_paths(artifact_root.parent)

    assert concrete.artifact_root == artifact_root
    assert parent.artifact_root == artifact_root
    assert concrete.model_path == artifact_root / "models" / "classifier_effnet_b0_best.pth"
    assert concrete.metrics_path == artifact_root / "reports" / "classifier_metrics.json"
    assert concrete.threshold_path == artifact_root / "reports" / "best_threshold.json"
    assert concrete.validation_predictions_path == artifact_root / "predictions" / "val_classifier_predictions.csv"


def test_resolve_artifact_root_reports_missing_required_files(tmp_path):
    from src.inference.predict import V1InferenceError, resolve_v1_artifact_paths

    artifact_root = _make_artifact_root(tmp_path)
    (artifact_root / "reports" / "classifier_metrics.json").unlink()

    with pytest.raises(V1InferenceError, match="classifier_metrics.json"):
        resolve_v1_artifact_paths(artifact_root)


def test_load_threshold_reads_saved_value_and_rejects_invalid(tmp_path):
    from src.inference.predict import V1InferenceError, load_threshold

    threshold_path = tmp_path / "best_threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 0.42}), encoding="utf-8")
    assert load_threshold(threshold_path) == pytest.approx(0.42)

    threshold_path.write_text(json.dumps({"threshold": 1.2}), encoding="utf-8")
    with pytest.raises(V1InferenceError, match="between 0.0 and 1.0"):
        load_threshold(threshold_path)


def test_predict_images_loads_checkpoint_and_applies_saved_threshold(tmp_path):
    from src.inference.predict import predict_images

    artifact_root = _make_artifact_root(tmp_path)
    image_paths = [
        _make_image(tmp_path / "images" / "img_a.jpg", (20, 20, 20)),
        _make_image(tmp_path / "images" / "img_b.jpg", (220, 220, 220)),
    ]

    predictions = predict_images(
        image_paths,
        artifact_root=artifact_root,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        batch_size=2,
    )

    assert [prediction.image_id for prediction in predictions] == ["img_a.jpg", "img_b.jpg"]
    assert [prediction.target for prediction in predictions] == [0, 0]
    assert all(prediction.probability == pytest.approx(0.5) for prediction in predictions)
    assert all(prediction.threshold == pytest.approx(0.6) for prediction in predictions)


def test_predict_images_does_not_create_out_of_scope_outputs(tmp_path):
    from src.inference.predict import predict_images

    artifact_root = _make_artifact_root(tmp_path)
    image_path = _make_image(tmp_path / "images" / "img.jpg")

    predict_images(
        [image_path],
        artifact_root=artifact_root,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
    )

    forbidden = [
        tmp_path / "outputs" / "hard_examples",
        tmp_path / "outputs" / "figures" / "gradcam.png",
        tmp_path / "outputs" / "predictions" / "test_predictions.csv",
        tmp_path / "outputs" / "submissions" / "submission_v2.csv",
    ]
    assert not any(path.exists() for path in forbidden)
