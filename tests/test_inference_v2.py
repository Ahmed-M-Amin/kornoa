"""Tests for SPEC-007 V2 inference, submission, and benchmark contracts."""

import csv
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.models.classifier import create_classifier


class _FakeVisionBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(8, 1000))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def _install_fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torchvision = types.ModuleType("torchvision")
    fake_models = types.SimpleNamespace(
        efficientnet_b0=lambda weights=None: _FakeVisionBackbone(),
        efficientnet_b1=lambda weights=None: _FakeVisionBackbone(),
        efficientnet_b2=lambda weights=None: _FakeVisionBackbone(),
        convnext_tiny=lambda weights=None: _FakeVisionBackbone(),
    )
    fake_torchvision.models = fake_models
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)


def _make_tiny_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = create_classifier(model_name="tiny_cnn", synthetic_smoke=True)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), path)
    return path


def _make_binary_checkpoint(path: Path, model_name: str = "tiny_cnn") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = create_classifier(model_name=model_name, synthetic_smoke=False)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), path)
    return path


def _make_v2_artifact_root(
    root: Path,
    *,
    model_name: str = "tiny_cnn",
    nested: bool = False,
    experiment_name: str = "v2b_effnet_b1",
) -> Path:
    artifact_root = root / "outputs" / "kaggle_v2"
    if nested:
        artifact_root = artifact_root / experiment_name / "kaggle_v2"
    if model_name == "tiny_cnn":
        _make_tiny_checkpoint(artifact_root / "models" / "classifier_best.pth")
    else:
        _make_binary_checkpoint(artifact_root / "models" / "classifier_best.pth", model_name)
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


def test_resolve_v2_artifact_root_supports_nested_experiment_parent(tmp_path):
    from src.inference.predict import resolve_v2_artifact_paths

    concrete_root = _make_v2_artifact_root(tmp_path, nested=True)
    experiment_root = concrete_root.parent
    paths = resolve_v2_artifact_paths(experiment_root)

    assert paths.artifact_root == concrete_root
    assert paths.model_path == concrete_root / "models" / "classifier_best.pth"
    assert paths.threshold_path == concrete_root / "reports" / "best_threshold.json"


def test_resolve_v2_artifact_root_supports_v2b_he_parent(tmp_path):
    from src.inference.predict import resolve_v2_artifact_paths

    concrete_root = _make_v2_artifact_root(tmp_path, nested=True, experiment_name="v2b_he_effnet_b1")
    paths = resolve_v2_artifact_paths(concrete_root.parent)

    assert paths.artifact_root == concrete_root
    assert paths.model_path == concrete_root / "models" / "classifier_best.pth"
    assert paths.threshold_path == concrete_root / "reports" / "best_threshold.json"


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


def test_efficientnet_b1_binary_checkpoint_loads_for_v2_prediction(tmp_path, monkeypatch):
    _install_fake_torchvision(monkeypatch)
    from src.inference.predict import predict_images

    artifact_root = _make_v2_artifact_root(tmp_path, model_name="efficientnet_b1")
    checkpoint = torch.load(artifact_root / "models" / "classifier_best.pth", map_location="cpu")
    assert checkpoint["backbone.classifier.1.weight"].shape[0] == 1
    dataset = _make_dataset(tmp_path)
    predictions = predict_images(
        [dataset / "test_images" / "img_a.jpg"],
        artifact_root=artifact_root,
        model_name="efficientnet_b1",
        device="cpu",
        image_size=64,
    )

    assert len(predictions) == 1
    assert predictions[0].target == 0


def test_efficientnet_b2_binary_checkpoint_loads_when_supported(tmp_path):
    pytest.importorskip("torchvision")
    from src.inference.predict import predict_images

    artifact_root = _make_v2_artifact_root(tmp_path, model_name="efficientnet_b2")
    dataset = _make_dataset(tmp_path)
    predictions = predict_images(
        [dataset / "test_images" / "img_a.jpg"],
        artifact_root=artifact_root,
        model_name="efficientnet_b2",
        device="cpu",
        image_size=64,
    )

    assert len(predictions) == 1
    assert predictions[0].target == 0


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


def test_v2_classifier_test_prediction_export_contract(tmp_path):
    from src.inference.submission import export_classifier_test_predictions

    artifact_root = _make_v2_artifact_root(tmp_path)
    dataset = _make_dataset(tmp_path)
    output_path = tmp_path / "outputs" / "hybrid" / "v4" / "input" / "test_classifier_predictions_v2b.csv"

    report = export_classifier_test_predictions(
        dataset_root=dataset,
        artifact_root=artifact_root,
        output_path=output_path,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        batch_size=2,
    )

    rows = list(csv.DictReader(output_path.open("r", newline="", encoding="utf-8")))
    assert report.row_count == 2
    assert list(rows[0]) == ["image_id", "prob_bad", "classifier_prediction", "target"]
    assert all(0.0 <= float(row["prob_bad"]) <= 1.0 for row in rows)
    assert {row["classifier_prediction"] for row in rows} <= {"0", "1"}
    assert [row["target"] for row in rows] == [row["classifier_prediction"] for row in rows]


def test_v2_submission_works_with_efficientnet_b1(tmp_path, monkeypatch):
    _install_fake_torchvision(monkeypatch)
    from src.inference.submission import generate_submission

    artifact_root = _make_v2_artifact_root(tmp_path, model_name="efficientnet_b1")
    dataset = _make_dataset(tmp_path)
    output_path = tmp_path / "outputs" / "kaggle_v2" / "submissions" / "submission_v2.csv"

    report = generate_submission(
        dataset_root=dataset,
        artifact_root=artifact_root,
        output_path=output_path,
        model_name="efficientnet_b1",
        device="cpu",
        image_size=64,
    )

    assert report.row_count == 2
    assert output_path.exists()


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


def test_v2_benchmark_works_with_efficientnet_b1(tmp_path, monkeypatch):
    _install_fake_torchvision(monkeypatch)
    from src.inference.benchmark import run_benchmark

    artifact_root = _make_v2_artifact_root(tmp_path, model_name="efficientnet_b1")
    dataset = _make_dataset(tmp_path)
    output_path = tmp_path / "outputs" / "kaggle_v2" / "benchmarks" / "v2_inference_benchmark.json"

    report = run_benchmark(
        image_dir=dataset / "test_images",
        artifact_root=artifact_root,
        output_path=output_path,
        model_name="efficientnet_b1",
        device="cpu",
        image_size=64,
        v1_average_time_per_image=10.0,
    )

    assert report.total_images == 2
    assert output_path.exists()
