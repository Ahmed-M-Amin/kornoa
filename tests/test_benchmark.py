"""Tests for SPEC-006 V1 inference benchmarking."""

from __future__ import annotations

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


def _make_image_dir(root: Path, count: int = 3) -> Path:
    image_dir = root / "test_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (48, 48), (100 + index, 100, 100)).save(image_dir / f"img_{index}.jpg")
    return image_dir


def test_run_benchmark_saves_required_v1_timing_report(tmp_path):
    from src.inference.benchmark import run_benchmark

    artifact_root = _make_artifact_root(tmp_path)
    image_dir = _make_image_dir(tmp_path)
    output_path = tmp_path / "outputs" / "benchmarks" / "v1_inference_benchmark.json"

    report = run_benchmark(
        image_dir=image_dir,
        artifact_root=artifact_root,
        output_path=output_path,
        model_name="tiny_cnn",
        synthetic_smoke=True,
        device="cpu",
        batch_size=2,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.total_images == 3
    assert saved["total_images"] == 3
    assert saved["total_time_seconds"] >= 0
    assert saved["average_time_per_image"] >= 0
    assert saved["images_per_second"] >= 0
    assert saved["batch_size"] == 2
    assert saved["device"] == "cpu"
    assert saved["model_name"] == "tiny_cnn"
    assert saved["image_size"] == 384
    assert saved["artifact_root"] == str(artifact_root)
    assert not (tmp_path / "outputs" / "hard_examples").exists()


def test_benchmark_cli_smoke(tmp_path):
    import subprocess
    import sys

    artifact_root = _make_artifact_root(tmp_path)
    image_dir = _make_image_dir(tmp_path, count=1)
    output_path = tmp_path / "outputs" / "benchmarks" / "v1_inference_benchmark.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.inference.benchmark",
            "--image-dir",
            str(image_dir),
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
