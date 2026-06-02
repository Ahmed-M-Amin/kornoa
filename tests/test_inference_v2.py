"""Tests for SPEC-007 V2 inference, submission, and benchmark contracts."""

import csv
import json
from pathlib import Path

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


def _make_v2_artifact_root(root: Path) -> Path:
    artifact_root = root / "outputs" / "kaggle_v2"
    _make_tiny_checkpoint(artifact_root / "models" / "classifier_best.pth")
    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "best_threshold.json").write_text(json.dumps({"threshold": 0.6}), encoding="utf-8")
    (reports / "classifier_metrics.json").write_text(json.dumps({"f1_score": 0.94}), encoding="utf-8")
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


def test_resolve_v2_artifact_root_uses_selected_checkpoint_name(tmp_path):
    from src.inference.predict import resolve_v2_artifact_paths

    artifact_root = _make_v2_artifact_root(tmp_path)
    paths = resolve_v2_artifact_paths(artifact_root)

    assert paths.model_path == artifact_root / "models" / "classifier_best.pth"
    assert paths.threshold_path == artifact_root / "reports" / "best_threshold.json"


def test_v2_prediction_uses_classifier_probability_and_threshold_only(tmp_path):
    from src.inference.predict import predict_images

    artifact_root = _make_v2_artifact_root(tmp_path)
    dataset = _make_dataset(tmp_path)
    predictions = predict_images(
        sorted((dataset / "test_images").glob("*.jpg")),
        artifact_root=artifact_root,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        batch_size=2,
    )

    assert [prediction.target for prediction in predictions] == [0, 0]
    assert not (tmp_path / "outputs" / "hard_examples").exists()


def test_v2_submission_output_contract(tmp_path):
    from src.inference.submission import generate_submission

    artifact_root = _make_v2_artifact_root(tmp_path)
    dataset = _make_dataset(tmp_path)
    output_path = tmp_path / "outputs" / "kaggle_v2" / "submissions" / "submission_v2.csv"

    report = generate_submission(
        dataset_root=dataset,
        artifact_root=artifact_root,
        output_path=output_path,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
    )

    rows = list(csv.DictReader(output_path.open("r", newline="", encoding="utf-8")))
    assert report.row_count == 2
    assert list(rows[0]) == ["image_id", "target"]
    assert {row["target"] for row in rows} <= {"0", "1"}


def test_v2_benchmark_report_includes_speed_multiplier(tmp_path):
    from src.inference.benchmark import run_benchmark

    artifact_root = _make_v2_artifact_root(tmp_path)
    dataset = _make_dataset(tmp_path)
    output_path = tmp_path / "outputs" / "kaggle_v2" / "benchmarks" / "v2_inference_benchmark.json"

    report = run_benchmark(
        image_dir=dataset / "test_images",
        artifact_root=artifact_root,
        output_path=output_path,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        v1_average_time_per_image=10.0,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.speed_multiplier_vs_v1 is not None
    assert saved["within_v2_speed_ceiling"] is True
    assert "speed_multiplier_vs_v1" in saved
