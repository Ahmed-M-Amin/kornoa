"""Tests for SPEC-010 V5 inference, export, and submission contracts."""

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.models.classifier import create_classifier
from src.training.train_classifier import validate_v5_test_predictions


def _make_tiny_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), path)
    return path


def _make_v5_artifact_root(root: Path) -> Path:
    artifact_root = root / "outputs" / "kaggle_v5" / "v5_strong_classifier"
    _make_tiny_checkpoint(artifact_root / "models" / "classifier_best.pth")
    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "best_threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.6,
                "f1_score": 0.92,
                "candidate_count": 101,
                "split_source": "v2b_compatible",
                "selection_source": "validation_only",
                "confusion_counts": {"tp": 1, "fp": 0, "tn": 1, "fn": 0},
            }
        ),
        encoding="utf-8",
    )
    (reports / "classifier_metrics.json").write_text(
        json.dumps(
            {
                "f1_score": 0.92,
                "threshold": 0.6,
                "validation_target_distribution": {"0": 1, "1": 1},
                "validation_prediction_distribution": {"0": 2, "1": 0},
                "test_prediction_distribution": {},
                "submission_row_count": 0,
                "v2b_benchmark_reference": {"reference_name": "v2b"},
                "public_score_baseline": 0.92181,
                "v2b_public_score": 0.92121,
                "v3_detector_public_score": 0.74169,
                "public_score_decision": "analysis_only_pending_manual_review",
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "predictions").mkdir(parents=True, exist_ok=True)
    (artifact_root / "predictions" / "val_classifier_predictions.csv").write_text(
        "image_id,true_label,prob_bad,classifier_prediction,target\nimg_a.jpg,0,0.2,0,0\nimg_b.jpg,1,0.3,0,0\n",
        encoding="utf-8",
    )
    return artifact_root


def _make_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    images = dataset / "test_images"
    images.mkdir(parents=True, exist_ok=True)
    for image_id in ["img_a.jpg", "img_b.jpg"]:
        Image.new("RGB", (48, 48), (120, 120, 120)).save(images / image_id)
    with (dataset / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        writer.writerow({"image_id": "img_a.jpg", "target": 0})
        writer.writerow({"image_id": "img_b.jpg", "target": 0})
    return dataset


def test_v5_test_prediction_export_and_submission_contracts(tmp_path):
    from src.inference.submission import export_classifier_test_predictions, generate_submission

    artifact_root = _make_v5_artifact_root(tmp_path)
    dataset = _make_dataset(tmp_path)
    test_predictions = artifact_root / "predictions" / "test_classifier_predictions.csv"
    submission_output = artifact_root / "submissions" / "submission_v5.csv"

    prediction_report = export_classifier_test_predictions(
        dataset_root=dataset,
        artifact_root=artifact_root,
        output_path=test_predictions,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        batch_size=2,
        image_size=64,
    )
    submission_report = generate_submission(
        dataset_root=dataset,
        artifact_root=artifact_root,
        output_path=submission_output,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        batch_size=2,
        image_size=64,
    )

    prediction_rows = list(csv.DictReader(test_predictions.open("r", newline="", encoding="utf-8")))
    submission_rows = list(csv.DictReader(submission_output.open("r", newline="", encoding="utf-8")))
    metrics = json.loads((artifact_root / "reports" / "classifier_metrics.json").read_text(encoding="utf-8"))

    assert prediction_report.row_count == 2
    assert submission_report.row_count == 2
    assert list(prediction_rows[0]) == ["image_id", "prob_bad", "classifier_prediction", "target"]
    assert all(0.0 <= float(row["prob_bad"]) <= 1.0 for row in prediction_rows)
    assert list(submission_rows[0]) == ["image_id", "target"]
    assert metrics["submission_row_count"] == 2
    assert metrics["test_prediction_distribution"] == {"0": 2, "1": 0}
    assert metrics["public_score_baseline"] == pytest.approx(0.92181)
    assert metrics["v3_detector_public_score"] == pytest.approx(0.74169)


def test_v5_test_prediction_validator_rejects_duplicates_out_of_range_and_labels(tmp_path):
    output_path = tmp_path / "bad.csv"
    output_path.write_text(
        "image_id,prob_bad,classifier_prediction,target,true_label\nimg_a.jpg,1.2,0,0,0\nimg_a.jpg,0.4,1,1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        validate_v5_test_predictions(output_path)


def test_v5_resolve_artifacts_and_scope_guards(tmp_path):
    from src.inference.predict import resolve_v5_artifact_paths

    artifact_root = _make_v5_artifact_root(tmp_path)
    paths = resolve_v5_artifact_paths(artifact_root)

    assert paths.model_path == artifact_root / "models" / "classifier_best.pth"
    assert paths.threshold_path == artifact_root / "reports" / "best_threshold.json"
    assert not (tmp_path / "outputs" / "hybrid" / "v4").exists()
    assert not (tmp_path / "outputs" / "detector").exists()
