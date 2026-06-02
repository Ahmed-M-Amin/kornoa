"""Tests for SPEC-006 V1 submission generation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
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
    (reports / "best_threshold.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    (reports / "classifier_metrics.json").write_text(json.dumps({"f1_score": 0.9}), encoding="utf-8")
    predictions = artifact_root / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    (predictions / "val_classifier_predictions.csv").write_text(
        "image_id,true_label,probability,threshold,predicted_label\n",
        encoding="utf-8",
    )
    return artifact_root


def _make_dataset(root: Path) -> Path:
    image_dir = root / "test_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (128, 128, 128)).save(image_dir / "bottle_b.jpg")
    Image.new("RGB", (48, 48), (128, 128, 128)).save(image_dir / "bottle_a.jpg")
    with (root / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        writer.writerow({"image_id": "bottle_b.jpg", "target": 0})
        writer.writerow({"image_id": "bottle_a.jpg", "target": 0})
    return root


def test_generate_submission_matches_sample_order_and_target_columns(tmp_path):
    from src.inference.submission import generate_submission

    dataset_root = _make_dataset(tmp_path / "dataset")
    artifact_root = _make_artifact_root(tmp_path)
    output_path = tmp_path / "outputs" / "submissions" / "submission_v1.csv"

    report = generate_submission(
        dataset_root=dataset_root,
        artifact_root=artifact_root.parent,
        output_path=output_path,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
    )

    rows = list(csv.DictReader(output_path.open("r", newline="", encoding="utf-8")))
    assert report.output_path == output_path
    assert report.row_count == 2
    assert rows == [
        {"image_id": "bottle_b.jpg", "target": "1"},
        {"image_id": "bottle_a.jpg", "target": "1"},
    ]
    assert list(rows[0].keys()) == ["image_id", "target"]
    assert not (tmp_path / "outputs" / "hard_examples").exists()


def test_generate_submission_fails_when_sample_image_missing(tmp_path):
    from src.inference.submission import SubmissionError, generate_submission

    dataset_root = _make_dataset(tmp_path / "dataset")
    (dataset_root / "test_images" / "bottle_a.jpg").unlink()
    artifact_root = _make_artifact_root(tmp_path)

    with pytest.raises(SubmissionError, match="bottle_a.jpg"):
        generate_submission(
            dataset_root=dataset_root,
            artifact_root=artifact_root,
            output_path=tmp_path / "outputs" / "submissions" / "submission_v1.csv",
            model_name="tiny_cnn",
            synthetic_smoke=True,
            device="cpu",
        )


def test_submission_output_is_ignored_when_under_outputs():
    from src.inference.submission import DEFAULT_SUBMISSION_OUTPUT
    from src.training.train_classifier import validate_generated_artifact_confidentiality

    confidentiality = validate_generated_artifact_confidentiality([DEFAULT_SUBMISSION_OUTPUT])

    assert confidentiality.tracked_paths == []
    assert confidentiality.ignored_paths == [DEFAULT_SUBMISSION_OUTPUT]


def test_submission_cli_smoke(tmp_path):
    dataset_root = _make_dataset(tmp_path / "dataset")
    artifact_root = _make_artifact_root(tmp_path)
    output_path = tmp_path / "outputs" / "submissions" / "submission_v1.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.submission",
            "--dataset-root",
            str(dataset_root),
            "--artifact-root",
            str(artifact_root),
            "--output-path",
            str(output_path),
            "--model-name",
            "tiny_cnn",
            "--synthetic-smoke",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
